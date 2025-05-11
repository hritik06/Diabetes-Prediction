import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64

# Load model
model = joblib.load('svm_model.pkl')

# Page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide", page_icon="🩺")

# Utility: round images
def make_image_round(image_path, size):
    image = Image.open(image_path).convert("RGBA")
    image = image.resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    round_image = Image.new("RGBA", (size, size))
    round_image.paste(image, (0, 0), mask=mask)
    return round_image

# Utility: to base64 for inline HTML
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Paths for About Us
professor_image_path = "./assets/prof.jpg"
team1_image_path     = "./assets/team1.jpg"
team2_image_path     = "./assets/team2.jpg"
team3_image_path     = "./assets/team3.jpg"

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Diabetes Prediction", "About Us"])

# ----------------- HOME -----------------
if page == "Home":
    st.title("🏠 Welcome to the Diabetes Prediction App")
    st.markdown("---")

    st.subheader("👨‍🔬 About the Project")
    st.write("""
    This application helps predict whether a person is diabetic or not using machine learning models.
    Along with the prediction, it also provides:
    - **Prediction Certainty**: Shows how confident the model is in its result.
    - **Target Weight Suggestion**: Recommends a target weight to help maintain a healthy BMI.
    """)

    st.markdown("---")



    st.subheader("📌 How to Use")
    st.write("""
    - Go to the **Diabetes Prediction** page from the sidebar.
    - Enter the required health details.
    - Get an instant prediction with certainty level.
    - Receive personalized BMI and weight suggestions for a healthier lifestyle.
    """)


# ----------------- DIABETES PREDICTION -----------------
elif page == "Diabetes Prediction":
    st.title("🩺 Diabetes Risk & BMI Advisor")
    st.markdown("---")

    # Prediction + recommendation logic
    max_safe_bmi   = 24.9
    min_risky_bmi  = 25.0

    def predict_and_recommend(input_data, height, smoking_history):
        arr   = np.array(input_data).reshape(1, -1)
        prob  = model.predict_proba(arr)[0][1]
        pred  = model.predict(arr)[0]
        cert  = prob if pred == 1 else (1 - prob)
        target_bmi    = min_risky_bmi if pred == 1 else max_safe_bmi
        target_weight = target_bmi * (height ** 2)

        if pred == 1:
            cat, color = "Diabetic", "Red"
            advice = (
                f"You are likely **diabetic** with **{cert*100:.2f}%** certainty.\n\n"
                f"- Recommended weight: **< {target_weight:.2f} kg** (BMI < {min_risky_bmi})"
            )
        else:
            cat, color = "Non-Diabetic", "Green"
            advice = (
                f"You are likely **non-diabetic** with **{cert*100:.2f}%** certainty.\n\n"
                f"- Keep weight below **{target_weight:.2f} kg**."
            )

        if smoking_history:
            advice += (
                "\n- **Quit smoking** to reduce health complications."
                if pred == 1
                else "\n- **Consider quitting smoking** for better health."
            )

        return cat, color, advice

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender  = st.selectbox("Gender", ["Male", "Female"])
            age     = st.number_input("Age", min_value=1, step=1)
        with col2:
            hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        with col3:
            smoking = st.selectbox("Smoking History", ["No", "Yes"])
            height  = st.number_input("Height (m)", min_value=1.0, max_value=2.5, step=0.01)
            bmi     = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        try:
            data = [
                1 if gender == "Male" else 0,
                age,
                1 if hypertension == "Yes" else 0,
                1 if heart_disease == "Yes" else 0,
                1 if smoking == "Yes" else 0,
                bmi
            ]
            cat, color, advice = predict_and_recommend(data, height, smoking == "Yes")

            st.markdown(f"<h3 style='color:{color};'>Prediction: {cat}</h3>", unsafe_allow_html=True)
            st.markdown("### 📝 Recommendation:")
            st.markdown(advice)
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")


# ----------------- ABOUT US -----------------
else:  # page == "About Us"
    st.title("About Us")
    st.header("Project Guide")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        prof_img        = make_image_round(professor_image_path, 150)
        prof_base64     = get_image_base64(prof_img)
        st.markdown(f"""
        <div class="card">
          <img src="data:image/png;base64,{prof_base64}">
          <div class="container">
            <b>Dr Lalu Seban</b><br>
            <a href="mailto:lalu@ei.nits.ac.in"><i class="fa fa-envelope"></i></a>
            <a href="http://eie.nits.ac.in/lalu/"><i class="fa fa-globe"></i></a>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.header("Our Team")
    team_members = [
        {"name":"Hritik Baranwal","path":team1_image_path,"email":"baranwalhritik@gmail.com","linkedin":"https://www.linkedin.com/in/hritik-baranwal-b65729237/","github":"https://github.com/hritik06"},
        {"name":"Vijay Kumar Kasaudhan","path":team2_image_path,"email":"kaushikborah4080@gmail.com","linkedin":"https://www.linkedin.com/in/kaushik-borah-317758226/","github":"https://github.com/dngeonMaster1706"},
        {"name":"Rahul Chauhan","path":team3_image_path,"email":"hritik21_ug@ei.nits.ac.in","linkedin":"https://www.linkedin.com/in/hritik-baranwal-b65729237/","github":"https://github.com/hritik06"}
    ]

    cols = st.columns(3)
    for idx, m in enumerate(team_members):
        img = make_image_round(m["path"], 100)
        b64 = get_image_base64(img)
        with cols[idx]:
            st.markdown(f"""
            <div class="card">
              <img src="data:image/png;base64,{b64}">
              <div class="container">
                <b>{m['name']}</b><br>
                <a href="mailto:{m['email']}"><i class="fa fa-envelope"></i></a>
                <a href="{m['linkedin']}"><i class="fa fa-linkedin"></i></a>
                <a href="{m['github']}"><i class="fa fa-github"></i></a>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Styles for cards & icons
    st.markdown("""
    <style>
      .card {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        width: 80%; margin: 10px auto; border-radius:10px; text-align:center; padding:10px;
      }
      .card img { border-radius:50%; width:50%; margin:10px 0; }
      .container { padding:2px 16px; }
      .fa { font-size:20px; margin:0 10px; }
      .fa:hover { color:#1e90ff; }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    """, unsafe_allow_html=True)
