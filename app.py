import streamlit as st

from utils.resume_parser import extract_resume_text
from utils.preprocessing import preprocess_text
from utils.matcher import analyze_match


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="ResumeX AI",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("📄 ResumeX AI")
st.subheader("NLP-Based Resume Analyzer & Job Matcher")

st.write(
    "Upload your resume and provide a job description to analyze "
    "your resume against the job requirements."
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ ResumeX AI")

    st.write(
        """
        ### How it works

        1. Upload your resume
        2. Enter the job description
        3. ResumeX extracts the text
        4. NLP preprocessing is performed
        5. TF-IDF calculates text similarity
        6. Skills are compared
        7. Missing skills are displayed
        """
    )

    st.divider()

    st.caption(
        "Built using Python, NLP, TF-IDF, Cosine Similarity and Streamlit."
    )


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    st.subheader("📄 Upload Resume")

    uploaded_resume = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
        help="Supported formats: PDF and DOCX"
    )


with col2:

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder=(
            "Example:\n\n"
            "We are looking for a Python developer with "
            "experience in SQL, Machine Learning, AWS and Docker."
        )
    )


st.divider()


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
)


if analyze_button:

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if uploaded_resume is None:

        st.error("❌ Please upload a PDF or DOCX resume.")

        st.stop()


    if not job_description.strip():

        st.error("❌ Please enter a job description.")

        st.stop()


    # -----------------------------------------------------
    # EXTRACT RESUME TEXT
    # -----------------------------------------------------

    try:

        resume_text = extract_resume_text(
            uploaded_resume
        )

    except Exception as e:

        st.error(f"❌ Could not read the resume: {e}")

        st.stop()


    if not resume_text.strip():

        st.error(
            "❌ No readable text was found in the resume."
        )

        st.stop()


    # -----------------------------------------------------
    # NLP PREPROCESSING
    # -----------------------------------------------------

    processed_resume = preprocess_text(
        resume_text
    )

    processed_job = preprocess_text(
        job_description
    )


    # -----------------------------------------------------
    # MATCHING
    # -----------------------------------------------------

    result = analyze_match(
        processed_resume,
        processed_job
    )


    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.success("✅ Resume analysis completed!")


    st.subheader("📊 Resume Match Score")


    score = result["match_score"]


    # Score display
    score_col1, score_col2, score_col3 = st.columns(3)


    with score_col1:

        st.metric(
            "Match Score",
            f"{score}%"
        )


    with score_col2:

        st.metric(
            "Matched Skills",
            len(result["matched_skills"])
        )


    with score_col3:

        st.metric(
            "Missing Skills",
            len(result["missing_skills"])
        )


    # Progress bar

    st.progress(
        min(score / 100, 1.0)
    )


    # -----------------------------------------------------
    # SKILL ANALYSIS
    # -----------------------------------------------------

    st.divider()

    st.subheader("🧠 Skill Analysis")


    skill_col1, skill_col2 = st.columns(2)


    with skill_col1:

        st.markdown("### ✅ Matched Skills")

        if result["matched_skills"]:

            for skill in result["matched_skills"]:

                st.success(
                    f"✓ {skill.title()}"
                )

        else:

            st.info(
                "No matching skills were detected."
            )


    with skill_col2:

        st.markdown("### ❌ Missing Skills")

        if result["missing_skills"]:

            for skill in result["missing_skills"]:

                st.error(
                    f"✗ {skill.title()}"
                )

        else:

            st.success(
                "🎉 No missing skills detected!"
            )


    # -----------------------------------------------------
    # ADDITIONAL SKILLS
    # -----------------------------------------------------

    st.divider()

    st.subheader("➕ Additional Resume Skills")


    if result["additional_skills"]:

        st.write(
            ", ".join(
                skill.title()
                for skill in result["additional_skills"]
            )
        )

    else:

        st.info(
            "No additional skills detected."
        )


    # -----------------------------------------------------
    # EXTRACTED RESUME TEXT
    # -----------------------------------------------------

    st.divider()

    with st.expander("📃 View Extracted Resume Text"):

        st.text_area(
            "Resume text",
            resume_text,
            height=300
        )


    # -----------------------------------------------------
    # TECHNICAL INFORMATION
    # -----------------------------------------------------

    with st.expander("🔬 NLP Processing Information"):

        st.write(
            """
            **Text Preprocessing**

            • Lowercase conversion  
            • URL removal  
            • Email removal  
            • Special-character removal  
            • Stopword removal  
            • Lemmatization  

            **Matching Technique**

            • TF-IDF Vectorization  
            • Cosine Similarity  

            **Skill Analysis**

            • Skill extraction  
            • Matched skill detection  
            • Missing skill detection
            """
        )