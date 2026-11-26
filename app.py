from datetime import datetime

import pandas as pd
import streamlit as st

from final_system import analyze_image

st.set_page_config(page_title="ROADGUARD AI", page_icon="◈", layout="wide")
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#07111F;--panel:#0D1B2A;--cyan:#50DFD0;--blue:#268BFF;--muted:#91A5B5}
.stApp{background:radial-gradient(ellipse at 80% -10%,#268bff20,transparent 38%),var(--bg);color:#eff7fa;font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
[data-testid="stSidebar"]{background:#091522;border-right:1px solid #7192a422}
.brand{font:700 1.2rem 'Space Grotesk';letter-spacing:.04em}.brand b{color:var(--cyan)}
.eyebrow{color:var(--cyan);font-size:.72rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase}
.hero{padding:1.6rem 0}.hero h1{font-size:clamp(2.8rem,5vw,4.8rem);line-height:1.02}.hero h1 em{color:var(--cyan);font-style:normal}
.hero p,.muted{color:var(--muted);line-height:1.7}.glass{background:linear-gradient(145deg,#112638,#0b1a28);border:1px solid #7192a422;border-radius:17px;padding:1.25rem;box-shadow:0 14px 40px #0002}
.art{height:280px;border:1px solid #50dfd033;border-radius:22px;position:relative;overflow:hidden;background:linear-gradient(160deg,#268bff22,transparent 55%),repeating-linear-gradient(0deg,transparent 0 35px,#50dfd018 36px),repeating-linear-gradient(90deg,transparent 0 35px,#50dfd012 36px),#0b1a28}
.art:after{content:'';position:absolute;width:160%;height:120px;left:-30%;bottom:-65px;transform:perspective(200px) rotateX(30deg);background:repeating-linear-gradient(90deg,transparent 0 50px,#50dfd088 51px 53px,transparent 54px 105px),#152b3d;box-shadow:0 -10px 50px #268bff25}
.scan{position:absolute;left:8%;width:84%;top:35%;height:1px;background:var(--cyan);box-shadow:0 0 20px var(--cyan);animation:scan 4s ease-in-out infinite alternate}@keyframes scan{to{top:75%;opacity:.3}}
.pill{color:var(--cyan);border:1px solid #50dfd044;background:#50dfd011;padding:.3rem .65rem;border-radius:40px;font-size:.75rem}
div[data-testid="stFileUploader"]{background:var(--panel);border:1px dashed #50dfd066;border-radius:16px;padding:1rem}
div.stButton>button{border:0;border-radius:10px;background:linear-gradient(110deg,var(--cyan),#63c9ff);color:#07111f;font-weight:700;transition:.2s}div.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 24px #50dfd044;color:#07111f}
[data-testid="stMetric"]{background:var(--panel);border:1px solid #7192a422;padding:1rem;border-radius:14px}hr{border-color:#7192a422}
</style>''', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "nav" not in st.session_state:
    st.session_state.nav = "Overview"

with st.sidebar:
    st.markdown('<div class="brand">◈ ROADGUARD <b>AI</b></div><small class="muted">INTELLIGENT ROADS · SAFER JOURNEYS</small>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "Road scanner", "Analytics", "About project"], key="nav")
    st.markdown("---")
    st.markdown('<span class="pill">● Local inference</span>', unsafe_allow_html=True)
    st.caption("YOLOv8 models · OpenCV illumination")
    st.caption("Records stay in this browser session.")

def goto(target):
    st.session_state.nav = target

def render_analysis(saved):
    result = saved["result"]
    st.markdown('<p class="eyebrow">Analysis complete</p>', unsafe_allow_html=True)
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Risk classification", result["risk"])
    x2.metric("Risk score · project rule", result["score"])
    x3.metric("Potholes", result["potholes"])
    x4.metric("Waste objects", result["waste"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Original image")
        st.image(saved["original"], use_container_width=True)
    with right:
        st.markdown("#### AI-annotated image")
        st.image(result["annotated_rgb"], use_container_width=True)
    light, conf, cls = st.columns(3)
    light.metric("Illumination", result["light"].title(), f"Mean brightness {result['brightness']:.1f} / 255")
    scores = [d["confidence"] for d in result["detections"]]
    conf.metric("Mean detection confidence", f"{sum(scores)/len(scores):.1%}" if scores else "—", f"Across {len(scores)} detections")
    cls.metric("Assessment score", result["score"], result["risk"])
    with st.expander("Detection details and risk factors"):
        st.caption("Existing rule: 2 points per pothole, 1 per waste object, +5 low light or +2 medium light; low ≤5, medium 6–10, high >10. This is a project heuristic, not validated accident probability.")
        st.write(f"Potholes: {result['potholes']*2} · Waste: {result['waste']} · Illumination: {5 if result['light']=='low' else 2 if result['light']=='medium' else 0}")
        if result["detections"]:
            st.dataframe(pd.DataFrame([{"Object":d["type"],"Confidence":f"{d['confidence']:.1%}"} for d in result["detections"]]), hide_index=True, use_container_width=True)
        else:
            st.info("The trained models returned no detections for this image.")

if page == "Overview":
    st.markdown('<div class="hero"><p class="eyebrow">Road intelligence, at a glance</p><h1>See the Risk.<br><em>Before It Happens.</em></h1><p>Turn a road image into a clear, explainable assessment. ROADGUARD AI combines trained object detectors with illumination analysis to surface potential road hazards.</p></div>', unsafe_allow_html=True)
    art, intro = st.columns([1.2,1], gap="large", vertical_alignment="center")
    with art:
        st.markdown('<div class="art"><div class="scan"></div></div>', unsafe_allow_html=True)
    with intro:
        st.markdown('<p class="eyebrow">From image to insight</p><h2>One image.<br>Three risk signals.</h2><p class="muted">Potholes, roadside waste and illumination — analyzed locally by the existing Python pipeline.</p>', unsafe_allow_html=True)
        st.button("Open road scanner →", type="primary", on_click=goto, args=("Road scanner",))
        st.button("Explore analytics ↗", on_click=goto, args=("Analytics",))
    st.markdown("### Built for the road ahead")
    a,b,c=st.columns(3)
    a.markdown('<div class="glass"><h3>⌁ Hazard detection</h3><p class="muted">Two trained YOLO models identify potholes and waste.</p></div>',unsafe_allow_html=True)
    b.markdown('<div class="glass"><h3>◉ Light intelligence</h3><p class="muted">OpenCV measures mean grayscale brightness.</p></div>',unsafe_allow_html=True)
    c.markdown('<div class="glass"><h3>⌗ Explainable scoring</h3><p class="muted">Review how each factor contributes to the existing project score.</p></div>',unsafe_allow_html=True)
    st.markdown("### How it works")
    st.markdown('<div class="glass">① Upload a road image &nbsp; → &nbsp; ② Existing YOLO weights detect hazards &nbsp; → &nbsp; ③ OpenCV measures illumination &nbsp; → &nbsp; ④ Existing rule returns a risk class</div>',unsafe_allow_html=True)

elif page == "Road scanner":
    st.markdown('<p class="eyebrow">Computer vision · Python inference</p><h1>Road hazard scanner</h1><p class="muted">JPG, JPEG or PNG. Predictions come from the existing trained models; no success is simulated.</p>',unsafe_allow_html=True)
    upload=st.file_uploader("Drop a road image here, or browse files",type=["jpg","jpeg","png"])
    if upload:
        data=upload.getvalue()
        st.image(data,caption=f"Ready to analyze · {upload.name}",width=720)
        if st.button("Analyze road image →",type="primary"):
            try:
                with st.spinner("Running pothole and waste models · measuring illumination…"):
                    result=analyze_image(data)
                st.session_state.last_result={"result":result,"original":data,"filename":upload.name}
                st.session_state.history.insert(0,{"analyzed_at":datetime.now().strftime("%Y-%m-%d %H:%M"),"filename":upload.name,"risk":result["risk"],"score":result["score"],"potholes":result["potholes"],"waste":result["waste"],"light":result["light"],"brightness":result["brightness"]})
                st.success("Analysis finished with the trained project models.")
            except (FileNotFoundError,ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Inference could not complete: {exc}")
                st.caption("Confirm project dependencies and trained weights are available, then retry.")
    if st.session_state.last_result:
        st.markdown("---")
        render_analysis(st.session_state.last_result)
    st.caption("Risk score follows final_system.py and is not a validated accident probability.")

elif page == "Analytics":
    st.markdown('<p class="eyebrow">Evidence, not estimates</p><h1>Risk analytics</h1><p class="muted">Only images analyzed in this browser session appear here. No demonstration figures are presented as live data.</p>',unsafe_allow_html=True)
    history=st.session_state.history
    if not history:
        st.markdown('<div class="glass"><h3>No recorded analyses yet</h3><p class="muted">Run an image through the scanner to build session analytics. Records clear when this session ends.</p></div>',unsafe_allow_html=True)
        st.button("Analyze your first image →", on_click=goto, args=("Road scanner",))
    else:
        df=pd.DataFrame(history)
        a,b,c,d=st.columns(4)
        a.metric("Images analyzed",len(df));b.metric("Potholes detected",int(df.potholes.sum()));c.metric("Waste objects",int(df.waste.sum()));d.metric("High-risk assessments",int((df.risk=="HIGH RISK").sum()))
        left,right=st.columns(2)
        with left:
            st.markdown("#### Risk distribution")
            st.bar_chart(df.risk.value_counts().reindex(["LOW RISK","MEDIUM RISK","HIGH RISK"],fill_value=0),color="#50dfd0")
        with right:
            st.markdown("#### Detections by image")
            st.bar_chart(df.set_index("analyzed_at")[["potholes","waste"]],color=["#50dfd0","#268bff"])
        st.markdown("#### Analysis history")
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.download_button("Download session report · CSV",df.to_csv(index=False).encode("utf-8"),"roadguard-analysis.csv","text/csv")
    st.markdown('<div class="glass"><h3>Future modules</h3><p class="muted">Hazard map · location-based reporting · PDF reports · admin dashboard · weather context</p><span class="pill">Planned · not operational</span></div>',unsafe_allow_html=True)

else:
    st.markdown('<p class="eyebrow">Project overview</p><h1>Safer journeys start with visibility.</h1><p class="muted">ROADGUARD AI explores how computer vision can surface road-condition signals from images.</p>',unsafe_allow_html=True)
    st.markdown('<div class="glass"><h3>Motivation</h3><p class="muted">Potholes, roadside waste and poor illumination can make roads harder to navigate. This project brings three visual signals into one review workflow.</p></div>',unsafe_allow_html=True)
    st.markdown("### System architecture")
    st.markdown('<div class="glass">Road image → Python / OpenCV decode → trained YOLOv8 pothole model + trained YOLO waste model → brightness thresholds → existing weighted risk rule → annotated image & assessment</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.markdown('<div class="glass"><h3>Models</h3><p class="muted">Existing weights: train2 pothole detector and train3 waste detector. This app does not retrain them.</p></div>',unsafe_allow_html=True)
    b.markdown('<div class="glass"><h3>Method</h3><p class="muted">Object counts contribute to a threshold-based score. Grayscale brightness adds the existing illumination adjustment.</p></div>',unsafe_allow_html=True)
    c.markdown('<div class="glass"><h3>Technology</h3><p class="muted">Python · Streamlit · Ultralytics YOLO · OpenCV · NumPy · Pandas</p></div>',unsafe_allow_html=True)
    st.markdown("### Interpretation")
    st.markdown('<p class="muted">The risk label is the output of this project’s rule-based heuristic. It is not a calibrated or validated measure of accident probability. Map reporting, persistent history, PDF export and weather analysis remain future scope.</p>',unsafe_allow_html=True)
