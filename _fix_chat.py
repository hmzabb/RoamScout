p=r"d:/zmz/project/chang_an_travel/chang_an_ai/app/routers/chat.py"
c=open(p,"r",encoding="utf-8").read()
svc_start=c.find("SERVICE_AGENT_PROMPT")
rag_start=c.find("async def _rag_frames")
block=open("d:/zmz/project/chang_an_travel/_service_block.py","r",encoding="utf-8").read()
c=c[:svc_start]+block+c[rag_start:]
open(p,"w",encoding="utf-8").write(c)
print("Done! Size:",len(c))