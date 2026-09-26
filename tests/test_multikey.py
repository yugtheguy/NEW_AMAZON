from pathlib import Path
import json
import pyarrow.parquet as pq

from amazon_er.config import load_config
from amazon_er.pipeline.multikey import run_multikey_retrieval
from amazon_er.pipeline.normalization import run_normalization

HEADER="entity_id\tbusiness_name\tbusiness_address\tcountry\n"

def dataset(root:Path):
    train,test=root/"train",root/"test"; train.mkdir(parents=True); test.mkdir()
    s1=["r1\tReliance Smart\t1 Central Road\tUS","g1\tGanesh Hardware\t42 MG Road\tUS","z1\tOdd Name\t77 Orchid Lane\tUS","n1\tAlpha Stores\t5 Harbor Market\tUS","p1\tOther Name\t8 Andheri Versova\tUS","t1\tCafé Monde\t9 Paris Street\tUS",*[f"c{i}\tCommon Shop{i}\t{i if i>1 else 42} Side Road\tUS" for i in range(1,5)],"india\tReliance Smart\t1 Central Road\tIndia"]
    s2=["S2-r\tReliance Retail\t99 Elsewhere\tUS","S2-g\tGanesh Traders\tShop 42 MG Rd\tUS","S2-z\tBad Business\t77 Orchid Place\tUS","S2-n\tAlpha Market\t100 Harbor Avenue\tUS","S2-p\tUnknown\t99 Andheri Versova\tUS","S2-t\tCafé Monde\tElsewhere\tUS","S2-common\tCommon Target\t42 Unknown\tUS","S2-cross\tReliance Smart\t1 Central Road\tUS"]
    s3=["S3-zero\tNo Match\tNo Place\tUS"]
    for source,rows in {"source1":s1,"source2":s2,"source3":s3}.items(): (train/f"train_{source}.tsv").write_text(HEADER+"\n".join(rows)+"\n",encoding="utf-8")
    for source in ("source1","source2","source3"): (test/f"test_{source}.tsv").write_text(HEADER+f"T-{source}\tTest\t1 Road\tUS\n")
    (train/"train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n"+"\n".join(["r1\tS2-r","g1\tS2-g","z1\tS2-z","n1\tS2-n","p1\tS2-p","t1\tS2-t","c1\tS2-common",*[f"c{i}\t" for i in range(2,5)],"india\t"]) +"\n",encoding="utf-8")
    return root

def test_multikey_expected_candidates_suppression_ranking_diagnostics_resume(tmp_path):
    cfg=load_config("configs/smoke.yaml"); data=dataset(tmp_path/"data"); normalized=tmp_path/"normalized"/"v1"; run_normalization(cfg,data_root=data,output_dir=normalized); output=tmp_path/"multikey"
    first=run_multikey_retrieval(cfg,normalized_root=normalized,data_root=data,output_dir=output,split="train",country="US",source="source2")
    rows=[]
    for path in output.rglob("part-*.parquet"): rows.extend(pq.ParquetFile(path).read().to_pylist())
    pairs={(r["target_entity_id"],r["candidate_s1_entity_id"]) for r in rows}
    for pair in (("S2-r","r1"),("S2-g","g1"),("S2-z","z1"),("S2-n","n1"),("S2-p","p1"),("S2-t","t1"),("S2-common","c1")): assert pair in pairs
    assert ("S2-cross","india") not in pairs
    common=next(r for r in rows if r["target_entity_id"]=="S2-common" and r["candidate_s1_entity_id"]=="c1")
    assert common["block_name_number_hit"] and not common["block_rare_name_hit"]
    ganesh=next(r for r in rows if r["target_entity_id"]=="S2-g" and r["candidate_s1_entity_id"]=="g1")
    assert ganesh["blocker_signal_count"]>1 and ganesh["blocker_rank"]==1
    translit=next(r for r in rows if r["target_entity_id"]=="S2-t" and r["candidate_s1_entity_id"]=="t1")
    assert translit["block_translit_name_hit"]
    assert first["raw_key_coverage"]==1.0 and first["pruned_pair_recall"]==1.0
    assert all(str(k) in first["ranked_recall"] for k in (1,3,5,10,20,50))
    manifests=list(output.rglob("part-*.manifest.json")); assert manifests
    second=run_multikey_retrieval(cfg,normalized_root=normalized,data_root=data,output_dir=output,split="train",country="US",source="source2")
    assert second["shards"]==first["shards"] and second["shards_skipped"]==second["shards"]
    assert len(rows)==len({(r["target_entity_id"],r["candidate_s1_entity_id"],r["target_source"]) for r in rows})
