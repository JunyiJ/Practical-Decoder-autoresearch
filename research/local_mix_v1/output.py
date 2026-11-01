"""Narrow per-file caps over exercised streaming publication; no launch authority."""
from pathlib import Path
import torch
from omegaconf import OmegaConf
from .evidence_contract import require,canonical
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import budgets
RAW_CAP=448*1024**2
EXPORT_CAP=112*1024**2
WEIGHT_CAP=RAW_CAP+EXPORT_CAP
TRAJECTORY_CAP=576*1024**2
METADATA_CAP=16*1024**2

class PerFileStream:
    def __init__(self,stream,cap):self.stream,self.cap=stream,cap
    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()
    def write(self,data):
        require(self.tell()+memoryview(data).nbytes<=self.cap,'Individual checkpoint cap exceeded')
        return self.stream.write(data)

class Sink:
    def __init__(self,writer,stage,slot,cfg):
        require(isinstance(writer,MetadataWriter),'Bounded metadata writer required')
        require(stage in ('pilots','quality'),'Training output stage required')
        self.writer,self.cfg=writer,cfg
        self.base=f'evidence/{stage}/{slot:02d}';self.root=writer.root/self.base
        require(sum(v for k,v in budgets().items() if k.startswith(self.base+'/'))<=METADATA_CAP,'Inherited per-trajectory metadata exceeds allowance')
        self.weights=CheckpointWriter(writer,stage,slot,WEIGHT_CAP)
        self.raw=None;self.export=None;self.stream=None;self.count=0
        t=cfg['training'];self.steps=sorted(set(range(0,t['max_iters'],t['eval_every']))|{t['max_iters']-1})
    def validate_config(self,cfg):
        require(canonical(OmegaConf.to_container(cfg,resolve=True))==canonical(self.cfg),'Sink config differs')
        require(self.cfg['training']['keep_best_checkpoint'] is False and self.cfg['training']['checkpoint_every']==0,'Final-only checkpoint policy required')
    def save(self,path,payload,cap):
        path=Path(path);require(path.parent==self.root/'checkpoints','Checkpoint path differs')
        stream=self.weights.open(path.name)
        try:
            torch.save(payload,PerFileStream(stream,cap))
            return stream.publish()
        finally:stream.close()
    def checkpoint(self,path,payload):
        path=Path(path)
        require(self.raw is None and path.name.startswith('final_') and payload['step']==self.cfg['training']['max_iters'],'One complete final raw checkpoint required')
        self.save(path,payload,RAW_CAP);self.raw=path
    def averaged(self,path,payload):
        path=Path(path);require(self.raw is not None and self.export is None and path.name=='h200.pt','One H200 export after raw required')
        require('optimizer_state' not in payload and 'averaged_weights' in payload,'Inference-only export required')
        self.save(path,payload,EXPORT_CAP);self.export=path
    def metric(self,path,value):
        require(Path(path)==self.root/'metrics.jsonl' and self.count<len(self.steps) and value['step']==self.steps[self.count],'Metric path or sequence differs')
        if self.stream is None:self.stream=self.writer.open_append(self.base+'/metrics.jsonl')
        self.stream.write((canonical(value)+'\n').encode());self.count+=1
    def summary(self,path,value):
        require(Path(path)==self.root/'summary.json','Summary path differs')
        require(self.count==len(self.steps),'Missing evaluation metrics')
        self.writer.publish_json(self.base+'/summary.json',value)
    def close(self):
        if self.stream is not None:self.stream.close()
