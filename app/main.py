import base64
import io
import secrets
from pathlib import Path
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import time
from uuid import uuid4
from app.config import settings
from app.agent import graph

ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title='Medical Agent Educational API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins.split(','),allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=4000)
    history: list[dict[str,str]] = Field(default_factory=list,max_length=10)
    image_data: str | None = None

def auth(authorization: str | None):
    if settings.api_token and not secrets.compare_digest(authorization or '',f'Bearer {settings.api_token}'):
        raise HTTPException(status_code=401,detail='Invalid API token')


def _groq_post(path, json_body=None, files=None, data=None, timeout=45):
    headers={'Authorization': f'Bearer {settings.groq_api_key}'}
    with httpx.Client(base_url=settings.groq_base_url, timeout=timeout) as client:
        r = client.post(path, headers=headers, json=json_body, files=files, data=data)
    if r.status_code != 200:
        raise HTTPException(502, 'AI service request failed')
    return r.json()


def inspect_image(data):
    """Validate and describe the attached image via the vision model.

    Returns (description, error). error is a short user-safe reason when the
    photo could not be analyzed (description is '' then). Raises 400 only for
    an invalid upload payload.
    """
    if not data:
        return '', None
    try:
        header,payload=data.split(',',1)
        mime=header.split(';')[0].split(':')[1]
        raw=base64.b64decode(payload,validate=True)
        if mime not in ('image/jpeg','image/png','image/webp') or len(raw)>5_000_000:
            raise ValueError('Only JPEG/PNG/WebP under 5MB supported')
    except (ValueError,IndexError) as exc:
        raise HTTPException(status_code=400,detail=f'Invalid image: {exc}')
    try:
        result=_groq_post('/chat/completions', json_body={'model':settings.groq_vision_model,'messages':[{'role':'user','content':[{'type':'text','text':'Describe only objectively visible medical-relevant features. Do not diagnose, estimate severity, or assert absence of illness. If image unclear, say so.'},{'type':'image_url','image_url':{'url':data}}]}],'max_tokens':300})
        description=result['choices'][0]['message']['content'] or ''
    except HTTPException:
        return '', 'the image analysis service did not respond'
    except (KeyError,IndexError,TypeError):
        return '', 'the image analysis service returned an unexpected response'
    if not description.strip():
        return '', 'the image analysis service returned no description'
    return description, None


@app.get('/',response_class=HTMLResponse)
def home():
    return (ROOT/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'status':'ok','configured':bool(settings.groq_api_key and settings.pinecone_api_key),'tavily':bool(settings.tavily_api_key)}



@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise HTTPException(503,'Set GROQ_API_KEY and PINECONE_API_KEY in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        t_total=time.perf_counter()
        t0=time.perf_counter()
        observation, image_error = inspect_image(payload.image_data)
        if payload.image_data:
            print(f"TIMING vision={time.perf_counter()-t0:.1f}s", flush=True)

        if payload.image_data and image_error:
            # Deterministic failure: no model answer, so nothing can claim to have seen the photo.
            return {
                'answer': ('Your image could not be analyzed this time (' + image_error + '). '
                           'Nothing was answered from the photo. Please try again, '
                           'or remove the image and ask your question as text only.'),
                'sources': [],
                'tools': [],
                'conversation_id': thread_id
            }

        state=graph.invoke(
            {
                'question':payload.message,
                'history':history,
                'image_description':observation,
                'kb':[],
                'web':[],
                'answer':'',
                'route':''
            },
            config=config
        )

        print(f"TIMING total={time.perf_counter()-t_total:.1f}s", flush=True)
        return {
            'answer':state['answer'],
            'sources': [{'type':'pdf','page':x['page'],'score':x['score']} for x in state['kb']]+[{'type':'web','url':x['url'],'title':x['title']} for x in state['web']],
            'tools': ['pinecone']+(['tavily'] if state['route']=='web' and settings.tavily_api_key else [])+(['vision'] if observation else []),
            'conversation_id':thread_id
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:',repr(exc),flush=True)
        raise HTTPException(502,'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.')


@app.post('/api/transcribe')
async def transcribe(file:UploadFile=File(...),authorization:str|None=Header(default=None)):
    auth(authorization)
    data=await file.read(10_000_001)
    if len(data)>10_000_000:
        raise HTTPException(413,'Audio exceeds 10 MB')
    try:
        result=_groq_post('/audio/transcriptions', files={'file':('voice.webm',data,'audio/webm')}, data={'model':settings.groq_stt_model}, timeout=60)
        return {'text':result.get('text','')}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(502,'Speech transcription unavailable')
