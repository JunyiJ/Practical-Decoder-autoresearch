"""V2 resource recorder adapters using enforced campaign metadata writers.

No CLI/launcher or numerical pass is provided here. A future worker must satisfy
all campaign gates and use cooperating bounded producers for its other outputs.
Checkpoint create/publish transitions must share the metadata namespace lock;
otherwise census races fail closed rather than silently skip artifacts.
"""
import json

from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from . import tokenizer_wide_hybrid_resources as resources


class _CampaignRecorder:
    STAGE=None
    SLOTS=0

    def __init__(self, writer, slot, plan_sha256, request_sha256):
        if not isinstance(writer,MetadataWriter) or type(slot) is not int or not 0<=slot<self.SLOTS:
            raise ValueError('Bounded writer and exact declared trajectory slot required')
        self.writer=writer
        self.stem=f'evidence/{self.STAGE}/{slot:02d}'
        self.event_bytes=0
        super().__init__(writer.root/self.stem,plan_sha256,request_sha256)

    def _open_trace(self):
        return self.writer.open_append(self.stem+'/resource-events.jsonl')

    def _emit(self,event,emergency=False):
        data=(json.dumps(event,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
        cap=resources.EVENT_LIMIT if emergency else resources.EVENT_LIMIT-4096
        if self.event_bytes+len(data)>cap:raise RuntimeError('Resource event log cap exceeded')
        # BoundedAppend checks aggregate group capacity, disk reserve, locks,
        # inode/size continuity and fsyncs independently of captured stdio.
        self.stream.write(data)
        self.event_bytes+=len(data)

    def _publish_report(self,report):
        self.writer.publish_json(self.stem+'/resource-audit.json',report)


class CampaignPilotResourceRecorder(_CampaignRecorder,resources.PilotResourceRecorder):
    STAGE='pilots'
    SLOTS=3


class CampaignQualityResourceRecorder(_CampaignRecorder,resources.QualityResourceRecorder):
    STAGE='quality'
    SLOTS=24
