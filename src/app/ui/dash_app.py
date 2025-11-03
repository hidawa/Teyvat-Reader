import dash
from dash import html, dcc, Input, Output, State
import base64
from src.utils.file_utils import save_upload_file
from src.models.request_models import ChatRequest
from src.agents.graph import run_agent
import asyncio

app = dash.Dash(__name__, title="Teyvat-Reader")

app.layout = html.Div([
    html.H2("Teyvat-Reader — Multimodal Chat"),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and Drop or ', html.A('Select an image')]),
        multiple=False
    ),
    html.Br(),
    dcc.Textarea(id='user-text', placeholder='質問や指示を日本語で入力', style={'width': '100%', 'height': 120}),
    html.Br(),
    html.Button('送信', id='submit-btn'),
    html.Hr(),
    html.Div(id='output')
])

# helper to decode file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    saved = save_upload_file(decoded, filename)
    return saved

@app.callback(
    Output('output', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('user-text', 'value'),
    prevent_initial_call=True
)
def handle_submit(n_clicks, contents, filename, user_text):
    image_path = None
    if contents and filename:
        image_path = parse_contents(contents, filename)

    # build request
    req = ChatRequest(text=user_text or "", image_path=image_path, reflection_rounds=2)

    # run agent synchronously via asyncio
    resp = asyncio.run(run_agent(req))
    # format response
    children = [html.H4("Answer"), html.Pre(resp.final_answer)]
    # children.append(html.H4("Steps"))
    # for s in resp.steps:
    #     children.append(html.Div([html.B(s.node), html.Pre(s.content)]))

    return children
