"""Audit witness linking both sealed selectors to the single test replay."""
import hashlib,json,time
from pathlib import Path
import heldout_value_e016 as e16
import encoder_value_e017 as e17
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
    e17.unopened();e16.verify();e17.verify()
    for folder in [e16.OUT,e17.OUT]:
        fit=e17.read(folder/'fit_manifest.json');e17.check_files(fit);assert fit['test_generations_existed_at_seal'] is False
        check=e17.read(folder/'prediction_verification.json');assert check['verified']
    p=e17.OUT/'test_start_witness.json';assert not p.exists()
    files=[Path(__file__).resolve(),e16.OUT/'fit_manifest.json',e16.OUT/'test_predictions.json',e17.OUT/'fit_manifest.json',e17.OUT/'predictions.json']
    e17.save(p,{'files':{str(f.relative_to(ROOT)):sha(f) for f in files},'started_unix_time':time.time(),'test_generations_existed_before_start':False})
    e16.run_split('test')
