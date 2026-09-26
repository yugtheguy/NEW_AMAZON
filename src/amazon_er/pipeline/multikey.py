"""Memory-bounded target-to-S1 multikey inverted blocking."""
from __future__ import annotations

import csv, json, math, os, subprocess, time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from amazon_er.config import config_hash
from amazon_er.data.paths import resolve_dataset_paths
from amazon_er.infra.artifacts import validate_artifact
from amazon_er.infra.hashing import hash_file, hash_mapping
from amazon_er.infra.manifests import ShardManifest, load_manifest
from amazon_er.infra.resources import ResourceMonitor
from amazon_er.paths import project_root, resolve_project_path
from amazon_er.pipeline.exact_structured import NormalizedShard, _discover_normalized

STAGE="multikey_blocker"
FAMILIES=("rare_name","name_number","number_address","name_address","address_pair","transliterated")
KEY_FAMILIES=("rare_name","name_number","number_address","name_address","address_pair","translit_name","translit_name_number")
CAP_NAMES={"rare_name":"rare_name_bucket_cap","name_number":"name_number_bucket_cap","number_address":"number_address_bucket_cap","name_address":"name_address_bucket_cap","address_pair":"address_pair_bucket_cap","translit_name":"translit_name_bucket_cap","translit_name_number":"translit_name_number_bucket_cap"}
FLAG_NAMES={"rare_name":"block_rare_name_hit","name_number":"block_name_number_hit","number_address":"block_number_address_hit","name_address":"block_name_address_hit","address_pair":"block_address_pair_hit","translit_name":"block_translit_name_hit","translit_name_number":"block_translit_name_number_hit"}
COLUMNS=("target_entity_id","candidate_s1_entity_id","target_source","country","blocker_hit",*FLAG_NAMES.values(),"blocker_signal_count","blocker_best_key_df","blocker_min_key_df","blocker_sum_idf","blocker_rank")

class MultikeyError(RuntimeError): pass

def _imports():
    try:
        import pyarrow as pa; import pyarrow.parquet as pq
    except ImportError as exc: raise RuntimeError("Multikey retrieval requires the data extra") from exc
    return pa,pq

def schema():
    pa,_=_imports()
    return pa.schema([*[pa.field(n,pa.string()) for n in COLUMNS[:4]],*[pa.field(n,pa.bool_()) for n in COLUMNS[4:12]],pa.field("blocker_signal_count",pa.uint8()),pa.field("blocker_best_key_df",pa.uint32()),pa.field("blocker_min_key_df",pa.uint32()),pa.field("blocker_sum_idf",pa.float32()),pa.field("blocker_rank",pa.uint8())])

def _commit():
    try: return subprocess.run(["git","-c",f"safe.directory={project_root()}","rev-parse","HEAD"],cwd=project_root(),capture_output=True,text=True,check=True,timeout=5).stdout.strip()
    except Exception: return os.environ.get("CODE_COMMIT","UNAVAILABLE")

def _atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); tmp.replace(path)

def _normal_tokens(value,min_len): return {x for x in str(value or "").split() if len(x)>=min_len}
def _numbers(value,limit): return list(dict.fromkeys(x for x in str(value or "").split("|") if x))[:limit]

@dataclass
class BlockIndex:
    n:int; postings:dict[tuple[str,str],list[int]]; df:dict[tuple[str,str],int]; ids:list[str]; entity_keys:dict[str,set[tuple[str,str]]]; name_df:Counter; address_df:Counter; translit_df:Counter

def _selected(tokens,df,max_df,limit): return sorted((x for x in tokens if 0<df.get(x,0)<=max_df),key=lambda x:(df[x],x))[:limit]

def _row_keys(row,index_dfs,cfg):
    name_df,address_df,translit_df=index_dfs
    names=_selected(_normal_tokens(row.get("name_core") or row.get("name_tokens"),int(cfg["min_name_token_length"])),name_df,int(cfg["rare_name_token_max_df"]),int(cfg["max_name_tokens_per_target"]))
    addresses=_selected(_normal_tokens(row.get("address_normalized"),int(cfg["min_address_token_length"])),address_df,int(cfg["rare_address_token_max_df"]),int(cfg["max_address_tokens_per_target"]))
    numbers=_numbers(row.get("numeric_tokens"),int(cfg["max_numeric_tokens_per_target"]))
    keys=set()
    for n in names: keys.add(("rare_name",n))
    for n in names:
        for number in numbers: keys.add(("name_number",n+"\x1f"+number))
    for number in numbers:
        for a in addresses: keys.add(("number_address",number+"\x1f"+a))
    for n in names:
        for a in addresses: keys.add(("name_address",n+"\x1f"+a))
    if len(addresses)>=2: keys.add(("address_pair","\x1f".join(sorted(addresses[:2]))))
    meaningful=any(ord(c)>127 for c in str(row.get("name_raw") or "")) or str(row.get("name_transliterated") or "")!=str(row.get("name_tokens") or "")
    if meaningful:
        translit=_selected(_normal_tokens(row.get("name_transliterated"),int(cfg["min_name_token_length"])),translit_df,int(cfg["rare_name_token_max_df"]),int(cfg["max_name_tokens_per_target"]))
        for n in translit: keys.add(("translit_name",n))
        for n in translit:
            for number in numbers: keys.add(("translit_name_number",n+"\x1f"+number))
    return keys

def build_index(paths,cfg):
    pa,pq=_imports(); cols=["entity_id","name_raw","name_tokens","name_core","name_transliterated","address_normalized","numeric_tokens"]
    table=pa.concat_tables([pq.ParquetFile(p).read(columns=cols) for p in paths]); rows=table.to_pylist()
    name_df=Counter(); address_df=Counter(); translit_df=Counter()
    for r in rows:
        name_df.update(_normal_tokens(r["name_core"] or r["name_tokens"],int(cfg["min_name_token_length"])))
        address_df.update(_normal_tokens(r["address_normalized"],int(cfg["min_address_token_length"])))
        translit_df.update(_normal_tokens(r["name_transliterated"],int(cfg["min_name_token_length"])))
    postings=defaultdict(list); entity_keys={}; ids=[]; dfs=(name_df,address_df,translit_df)
    for i,r in enumerate(rows):
        entity=str(r["entity_id"]); ids.append(entity); keys=_row_keys(r,dfs,cfg); entity_keys[entity]=keys
        for key in keys: postings[key].append(i)
    postings=dict(postings); return BlockIndex(len(ids),postings,{k:len(v) for k,v in postings.items()},ids,entity_keys,name_df,address_df,translit_df)

def retrieve(row,index,cfg):
    keys=_row_keys(row,(index.name_df,index.address_df,index.translit_df),cfg); evidence=defaultdict(dict); suppressed=[]
    for key in keys:
        family=key[0]; bucket=index.postings.get(key,()); df=len(bucket); cap=int(cfg[CAP_NAMES[family]])
        if df>cap: suppressed.append((family,df)); continue
        for candidate in bucket: evidence[candidate][key]=df
    ranked=[]
    for candidate,items in evidence.items():
        dfs=list(items.values()); flags={FLAG_NAMES[f]:False for f in KEY_FAMILIES}
        for family,_ in items: flags[FLAG_NAMES[family]]=True
        ranked.append((candidate,sum(math.log((index.n+1)/(df+1)) for df in dfs),len(items),min(dfs),flags))
    ranked.sort(key=lambda x:(-x[1],-x[2],index.ids[x[0]])); raw_count=len(ranked); ranked=ranked[:int(cfg["blocker_max_candidates_per_target"])]
    output=[]
    for rank,(candidate,weight,count,min_df,flags) in enumerate(ranked,1):
        output.append({"target_entity_id":str(row["entity_id"]),"candidate_s1_entity_id":index.ids[candidate],"blocker_hit":True,**flags,"blocker_signal_count":count,"blocker_best_key_df":min_df,"blocker_min_key_df":min_df,"blocker_sum_idf":weight,"blocker_rank":rank})
    return output,keys,suppressed,raw_count

def _gt_map(path,s1_ids):
    result={}
    with path.open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream,delimiter="\t"):
            if row["source1_entity_id"] not in s1_ids: continue
            for target in row["matched_entity_ids"].split(","):
                if target.strip(): result[target.strip()]=row["source1_entity_id"]
    return result

def _artifact_paths(root,target):
    directory=root/target.split/f"country={quote(target.country,safe='')}"/f"source={target.source}"; artifact=directory/f"part-{int(target.shard_id):05d}.parquet"; return artifact,artifact.with_suffix(".manifest.json"),artifact.with_suffix(".stats.json")

def validate(path,country,source,expected):
    _,pq=_imports(); pf=pq.ParquetFile(path)
    if pf.metadata.num_rows!=expected or pf.schema_arrow!=schema(): raise ValueError("multikey schema/count mismatch")
    seen=set()
    for batch in pf.iter_batches(columns=list(COLUMNS[:4])):
        for r in batch.to_pylist():
            key=(r["target_entity_id"],r["candidate_s1_entity_id"],r["target_source"])
            if r["country"]!=country or r["target_source"]!=source or key in seen: raise ValueError("multikey identity/uniqueness failure")
            seen.add(key)

def _dependency(s1,target,cfg): return hash_mapping({"normalization":"normalization_v1","inputs":[(x.artifact_hash,x.input_fingerprint) for x in [*s1,target]],"blocker":cfg["retrieval"]["multikey"]})

def _valid(artifact,manifest_path,stats_path,target,dependency,cfg,commit):
    if not manifest_path.exists() or not stats_path.exists(): return False
    try:
        m=load_manifest(manifest_path); stats=json.loads(stats_path.read_text())
        if m.code_commit!=commit or stats["dependency"]!=dependency or Path(m.artifact_path).resolve()!=artifact.resolve(): return False
        return validate_artifact(manifest_path,expected_config_hash=config_hash(cfg),expected_stage=STAGE,expected_stage_version=str(cfg["retrieval"]["multikey"]["version"]),expected_input_fingerprint=dependency,schema_validator=lambda p:validate(p,target.country,target.source.replace("source","S"),m.output_rows)).valid
    except Exception:return False

def _process(index,target,s1,root,cfg,commit,truth):
    pa,pq=_imports(); artifact,manifest_path,stats_path=_artifact_paths(root,target); dep=_dependency(s1,target,cfg)
    if _valid(artifact,manifest_path,stats_path,target,dep,cfg,commit): return json.loads(stats_path.read_text()),True
    artifact.parent.mkdir(parents=True,exist_ok=True); tmp=artifact.with_suffix(".parquet.tmp"); writer=pq.ParquetWriter(tmp,schema(),compression="zstd"); started=time.monotonic(); out_rows=0
    stats={"dependency":dep,"targets":0,"candidate_hist":{},"truncated_targets":0,"suppressed":{},"largest_bucket":{},"gt_total":0,"raw_covered":0,"pruned_recovered":0,"lost_all_shared_suppressed":0,"family_recovered":{f:0 for f in FAMILIES},"family_unique":{f:0 for f in FAMILIES},"rank_hits":{str(k):0 for k in (1,3,5,10,20,50)}}
    columns=["entity_id","name_raw","name_tokens","name_core","name_transliterated","address_normalized","numeric_tokens"]
    for batch in pq.ParquetFile(target.path).iter_batches(batch_size=int(cfg["retrieval"]["multikey"]["target_batch_rows"]),columns=columns):
        output=[]
        for row in batch.to_pylist():
            candidates,keys,suppressed,raw_count=retrieve(row,index,cfg["retrieval"]["multikey"]); stats["targets"]+=1; stats["candidate_hist"][str(len(candidates))]=stats["candidate_hist"].get(str(len(candidates)),0)+1; stats["truncated_targets"]+=raw_count>len(candidates)
            for family,size in suppressed: stats["suppressed"][family]=stats["suppressed"].get(family,0)+1; stats["largest_bucket"][family]=max(stats["largest_bucket"].get(family,0),size)
            true=truth.get(str(row["entity_id"]))
            if true:
                stats["gt_total"]+=1; shared=keys & index.entity_keys[true]; raw=bool(shared); stats["raw_covered"]+=raw
                pruned={key for key in shared if index.df[key]<=int(cfg["retrieval"]["multikey"][CAP_NAMES[key[0]]])}; stats["pruned_recovered"]+=bool(pruned); stats["lost_all_shared_suppressed"]+=raw and not pruned
                family_hits={f:False for f in FAMILIES}
                for key in pruned: family_hits["transliterated" if key[0].startswith("translit") else key[0]]=True
                earlier=False
                for family in FAMILIES: stats["family_recovered"][family]+=family_hits[family]; stats["family_unique"][family]+=family_hits[family] and not earlier; earlier|=family_hits[family]
                true_rank=next((x["blocker_rank"] for x in candidates if x["candidate_s1_entity_id"]==true),None)
                for k in (1,3,5,10,20,50): stats["rank_hits"][str(k)]+=true_rank is not None and true_rank<=k
            for candidate in candidates: candidate["target_source"]=target.source.replace("source","S"); candidate["country"]=target.country
            output.extend(candidates)
        if output: table=pa.Table.from_pylist(output,schema=schema()); writer.write_table(table); out_rows+=table.num_rows
    writer.close(); tmp.replace(artifact); validate(artifact,target.country,target.source.replace("source","S"),out_rows); sample=ResourceMonitor(STAGE,country=target.country,source=target.source,shard=target.shard_id).sample(target.rows); stats["runtime_seconds"]=time.monotonic()-started; stats["peak_ram_bytes"]=sample.rss_bytes; stats["available_ram_bytes"]=sample.available_ram_bytes; stats["ram_fraction"]=sample.ram_fraction
    ShardManifest.complete(stage=STAGE,stage_version=str(cfg["retrieval"]["multikey"]["version"]),split=target.split,country=target.country,source=target.source.replace("source","S"),shard_id=target.shard_id,input_rows=target.rows,output_rows=out_rows,runtime_seconds=stats["runtime_seconds"],peak_ram_bytes=sample.rss_bytes,peak_vram_bytes=None,config_hash=config_hash(cfg),code_commit=commit,input_fingerprint=dep,artifact_path=str(artifact.resolve()),artifact_size_bytes=artifact.stat().st_size,artifact_hash=hash_file(artifact)).write(manifest_path); _atomic(stats_path,stats); return stats,False

def _aggregate(items):
    total={"targets":0,"candidate_hist":Counter(),"truncated_targets":0,"gt_total":0,"raw_covered":0,"pruned_recovered":0,"lost_all_shared_suppressed":0,"family_recovered":Counter(),"family_unique":Counter(),"rank_hits":Counter(),"suppressed":Counter(),"largest_bucket":{},"runtime_seconds":0.0,"peak_ram_bytes":0}
    breakdown={}; split_hist=defaultdict(Counter)
    for identity,s in items:
        for name in ("targets","truncated_targets","gt_total","raw_covered","pruned_recovered","lost_all_shared_suppressed"): total[name]+=s[name]
        for name in ("candidate_hist","family_recovered","family_unique","rank_hits","suppressed"): total[name].update(s[name])
        split_hist[identity.split()[0]].update(s["candidate_hist"])
        for f,v in s["largest_bucket"].items(): total["largest_bucket"][f]=max(total["largest_bucket"].get(f,0),v)
        total["runtime_seconds"]+=s["runtime_seconds"]; total["peak_ram_bytes"]=max(total["peak_ram_bytes"],s["peak_ram_bytes"])
        if s["gt_total"]: label=" ".join(identity.split()[1:]); breakdown.setdefault(label,{"gt_total":0,"raw":0,"pruned":0,"rank":Counter()}); b=breakdown[label]; b["gt_total"]+=s["gt_total"]; b["raw"]+=s["raw_covered"]; b["pruned"]+=s["pruned_recovered"]; b["rank"].update(s["rank_hits"])
    hist=total["candidate_hist"]
    def volume(h):
        total_count=sum(h.values())
        def at(p):
            threshold=int((total_count-1)*p)+1; cumulative=0
            for n,c in sorted((int(k),v) for k,v in h.items()):
                cumulative+=c
                if cumulative>=threshold:return n
            return 0
        return {"average":sum(n*c for n,c in ((int(k),v) for k,v in h.items()))/sum(h.values()) if h else 0,"p50":at(.5),"p90":at(.9),"p95":at(.95),"p99":at(.99),"max":max((int(k) for k in h),default=0)}
    summary={**{k:v for k,v in total.items() if k!="candidate_hist"},"status":"complete","raw_key_coverage":total["raw_covered"]/total["gt_total"] if total["gt_total"] else None,"pruned_pair_recall":total["pruned_recovered"]/total["gt_total"] if total["gt_total"] else None,"ranked_recall":{k:v/total["gt_total"] for k,v in total["rank_hits"].items()} if total["gt_total"] else {},"candidate_volume":volume(hist),"candidate_volume_by_split":{k:volume(v) for k,v in split_hist.items()},"breakdown":{k:{"gt_total":v["gt_total"],"raw_key_coverage":v["raw"]/v["gt_total"],"pruned_pair_recall":v["pruned"]/v["gt_total"],"ranked_recall":{rk:rv/v["gt_total"] for rk,rv in v["rank"].items()}} for k,v in breakdown.items()}}
    return summary

def run_multikey_retrieval(cfg,*,normalized_root,data_root=None,output_dir=None,split=None,source=None,country=None,shard_id=None):
    if normalized_root is None: raise MultikeyError("--normalized-root is required")
    source_map={None:None,"source2":"source2","source3":"source3","S2":"source2","S3":"source3"}; selected=source_map.get(source,"INVALID")
    if split not in {None,"train","test"} or selected=="INVALID": raise MultikeyError("invalid split/source")
    shard=None if shard_id is None else str(int(shard_id)); inventory=_discover_normalized(Path(normalized_root).resolve()); targets=[x for x in inventory if x.source in {"source2","source3"} and (split is None or x.split==split) and (selected is None or x.source==selected) and (country is None or x.country==country) and (shard is None or x.shard_id==shard)]
    if not targets: raise MultikeyError("no target shards match")
    root=Path(output_dir).resolve() if output_dir else resolve_project_path(cfg,"artifact_root")/"retrieval"/str(cfg["retrieval"]["version"])/STAGE; report_root=resolve_project_path(cfg,"artifact_root")/"reports"/"retrieval"/str(cfg["retrieval"]["version"])/STAGE; report_root.mkdir(parents=True,exist_ok=True); resource_log=report_root/"resources.jsonl"; paths=resolve_dataset_paths(cfg,data_root) if any(t.split=="train" for t in targets) else None; results=[]; commit=_commit(); skipped_count=0
    for current_split,current_country in sorted({(t.split,t.country) for t in targets}):
        s1=[x for x in inventory if x.split==current_split and x.source=="source1" and x.country==current_country]; print(f"[MULTIKEY] build index split={current_split} country={current_country}",flush=True); index=build_index([x.path for x in s1],cfg["retrieval"]["multikey"]); truth=_gt_map(paths.train_ground_truth,set(index.ids)) if current_split=="train" else {}
        for target in [x for x in targets if (x.split,x.country)==(current_split,current_country)]:
            print(f"[MULTIKEY] split={target.split} country={target.country} source={target.source} shard={target.shard_id}",flush=True); stats,skipped=_process(index,target,s1,root,cfg,commit,truth); skipped_count+=skipped; results.append((f"{target.split} {target.country} {target.source.replace('source','S')}",stats)); print(f"[MULTIKEY] complete skipped={skipped} candidates={sum(int(k)*v for k,v in stats['candidate_hist'].items())} rss={stats['peak_ram_bytes']}",flush=True)
            with resource_log.open("a",encoding="utf-8") as stream: stream.write(json.dumps({"split":target.split,"country":target.country,"source":target.source,"shard":target.shard_id,"rss_bytes":stats["peak_ram_bytes"],"available_ram_bytes":stats.get("available_ram_bytes"),"ram_fraction":stats.get("ram_fraction"),"runtime_seconds":stats["runtime_seconds"]},sort_keys=True)+"\n")
            if float(stats.get("ram_fraction",0))>=float(cfg["runtime"]["critical_ram_fraction"]): raise MultikeyError("critical RAM threshold reached after safe shard completion")
        del index,truth
    summary=_aggregate(results); summary["output_root"]=str(root); summary["shards"]=len(results); summary["shards_skipped"]=skipped_count; _atomic(report_root/"summary.json",summary); lines=["# Multikey Blocker Summary","",f"Raw key coverage: {summary['raw_key_coverage']}",f"Pruned pair recall: {summary['pruned_pair_recall']}","",json.dumps(summary,indent=2,sort_keys=True)]; report_root.mkdir(parents=True,exist_ok=True); (report_root/"summary.md").write_text("\n".join(lines),encoding="utf-8"); return summary
