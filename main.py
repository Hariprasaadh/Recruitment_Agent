from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import PyPDF2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import json

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # App-specific password for Gmail
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_retries=2,
)

# Sample job description
JOB_DESCRIPTION = """
Python Developer Position
Requirements:
- 5+ years of professional Python development experience
- Strong proficiency in Python 3.x, Django/Flask, and SQL databases
- Experience with RESTful APIs and microservices architecture
- Familiarity with cloud platforms (AWS/GCP/Azure)
- Strong problem-solving skills and unit testing experience
- Excellent communication skills
Preferred:
- Experience with Docker and Kubernetes
- Knowledge of CI/CD pipelines
- Bachelor's degree in Computer Science or related field
"""

# Define GraphState
class GraphState(TypedDict):
    application: str
    job_description: str
    candidate_email: str
    experience_level: str
    skill_match: str
    response: str
    extracted_resume: str

def extract_resume_text(pdf_file_path: str) -> str:
    """Extract text from a PDF resume."""
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Send an email to the candidate with the screening response."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def categorize_experience(state: GraphState) -> GraphState:
    """Categorize candidate's experience level based on resume and job description."""
    print("\nCategorizing experience level of candidate")
    prompt = ChatPromptTemplate.from_template(
        """Given the job description and candidate's resume, determine the candidate's experience level.
        Job Description: {job_description}
        Resume: {application}
        
        Analyze years of experience, roles, and responsibilities to categorize as:
        - Entry Level (0-2 years)
        - Mid Level (3-5 years)
        - Senior Level (6+ years)
        
        Provide a brief explanation for your categorization and return only the level (Entry Level, Mid Level, or Senior Level)."""
    )
    
    chain = prompt | llm
    experience_level = chain.invoke({
        "job_description": state["job_description"],
        "application": state["extracted_resume"]
    }).content.strip().split('\n')[-1]
    return {"experience_level": experience_level}

def assess_skills(state: GraphState) -> GraphState:
    """Assess candidate's skills against job requirements."""
    print("\nAssessing candidate skills")
    prompt = ChatPromptTemplate.from_template(
        """Compare the candidate's resume with the job description and assess skill compatibility.
        Job Description: {job_description}
        Resume: {application}
        
        Evaluate based on:
        - Required technical skills (Python, frameworks, databases, etc.)
        - Preferred skills
        - Relevant project experience
        - Education qualifications
        
        Provide a brief explanation and return 'Match' if the candidate meets 80%+ of requirements,
        'Partial Match' if 50-79% of requirements are met, or 'No Match' if below 50%.
        Return only the match level (Match, Partial Match, or No Match)."""
    )
    
    chain = prompt | llm
    skill_match = chain.invoke({
        "job_description": state["job_description"],
        "application": state["extracted_resume"]
    }).content.strip().split('\n')[-1]
    return {"skill_match": skill_match}

def schedule_interview(state: GraphState) -> GraphState:
    """Schedule an interview and notify candidate via email."""
    print("\nScheduling interview with candidate")
    response = "Candidate has been shortlisted for an interview. Recommended for technical interview round."
    email_body = f"""Dear Candidate,

Thank you for applying for the Python Developer position. Based on our review of your resume, we are pleased to invite you for a technical interview.

Next Steps:
- Our HR team will contact you within 2-3 business days to schedule the interview.
- Please prepare to discuss your experience with Python, Django/Flask, and cloud platforms.

Best regards,
Hiring Team
"""
    send_email(state["candidate_email"], "Interview Invitation - Python Developer Position", email_body)
    return {'response': response}

def escalate_to_recruiter(state: GraphState) -> GraphState:
    """Escalate to a recruiter for senior candidates and notify candidate."""
    print("\nEscalating to a recruiter")
    response = "Candidate has been escalated to a recruiter for senior-level review."
    email_body = f"""Dear Candidate,

Thank you for applying for the Python Developer position. Your application has been escalated to our senior recruitment team for further review due to your extensive experience.

Next Steps:
- A senior recruiter will reach out within 3-5 business days to discuss potential opportunities.
- Please feel free to contact us if you have any questions.

Best regards,
Hiring Team
"""
    send_email(state["candidate_email"], "Application Update - Python Developer Position", email_body)
    return {'response': response}

def reject_application(state: GraphState) -> GraphState:
    """Reject the application and notify candidate."""
    print("\nRejecting application")
    response = "Candidate does not meet the minimum requirements for the position."
    email_body = f"""Dear Candidate,

Thank you for applying for the Python Developer position. After careful consideration, we have determined that your qualifications do not fully meet the requirements for this role.

We appreciate your interest and encourage you to apply for other positions that align with your skills and experience.

Best regards,
Hiring Team
"""
    send_email(state["candidate_email"], "Application Update - Python Developer Position", email_body)
    return {'response': response}

# Initialize workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("extract_resume", lambda state: {"extracted_resume": extract_resume_text(state["application"])})
workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skills)
workflow.add_node("schedule_interview", schedule_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)

def route_func(state: GraphState) -> str:
    """Route based on skill match and experience level."""
    if state['skill_match'] == 'Match':
        return 'schedule_interview'
    elif state['skill_match'] == 'Partial Match' and state['experience_level'] == 'Senior Level':
        return 'escalate_to_recruiter'
    else:
        return 'reject_application'

# Define edges
workflow.add_edge(START, "extract_resume")
workflow.add_edge("extract_resume", "categorize_experience")
workflow.add_edge("categorize_experience", "assess_skillset")
workflow.add_conditional_edges("assess_skillset", route_func)
workflow.add_edge("schedule_interview", END)
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)

# Compile workflow
app = workflow.compile()

def run_candidate_screening(pdf_path: str, candidate_email: str, job_description: str = JOB_DESCRIPTION):
    """Run the candidate screening process with a PDF resume and send email notification."""
    results = app.invoke({
        "application": pdf_path,
        "job_description": job_description,
        "candidate_email": candidate_email
    })
    return {
        "resume_path": pdf_path,
        "candidate_email": candidate_email,
        "extracted_resume": results["extracted_resume"],
        "experience_level": results["experience_level"],
        "skill_match": results["skill_match"],
        "response": results["response"]
    }


st.set_page_config(
    page_title="AI Candidate Screening System",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        text-align: center;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .status-match {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-partial {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-no-match {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .processing-animation {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = []
if 'current_screening' not in st.session_state:
    st.session_state.current_screening = None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Candidate Screening System</h1>
        <p>Intelligent resume screening and candidate evaluation powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        
        # Job Description
        st.markdown("### 📋 Job Description")
        job_description = st.text_area(
            "Job Requirements",
            value=JOB_DESCRIPTION,
            height=300,
            help="Enter the job description and requirements"
        )
        
        # Statistics
        if st.session_state.screening_results:
            st.markdown("### 📊 Statistics")
            display_statistics()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🏠 Screen Candidates", "📈 Dashboard", "📋 History"])
    
    with tab1:
        screening_interface(job_description)
    
    with tab2:
        dashboard_interface()
    
    with tab3:
        history_interface()

def screening_interface(job_description):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload the candidate's resume in PDF format"
        )
        
        # Candidate email
        candidate_email = st.text_input(
            "📧 Candidate Email",
            placeholder="candidate@example.com",
            help="Email address to send screening results"
        )
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            send_email_notification = st.checkbox("Send Email Notification", value=True)
            auto_process = st.checkbox("Auto-process on upload", value=False)
        
        with col_opt2:
            experience_threshold = st.selectbox(
                "Experience Level Filter",
                ["All Levels", "Entry Level", "Mid Level", "Senior Level"]
            )
            
            skill_threshold = st.selectbox(
                "Minimum Skill Match",
                ["Any Match", "Partial Match", "Full Match"]
            )
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        # Process button
        if st.button("🚀 Process Resume", type="primary", use_container_width=True):
            if uploaded_file and candidate_email:
                process_resume(uploaded_file, candidate_email, job_description, send_email_notification)
            else:
                st.error("Please upload a resume and enter candidate email")
        
        # Clear results
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.screening_results = []
            st.session_state.current_screening = None
            st.rerun()
        
        # Export results
        if st.session_state.screening_results:
            if st.button("📥 Export Results", use_container_width=True):
                export_results()
    
    # Display current screening result
    if st.session_state.current_screening:
        display_screening_result(st.session_state.current_screening)

def process_resume(uploaded_file, candidate_email, job_description, send_email):
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with st.spinner("Processing resume..."):
            # Save file
            status_text.text("📁 Saving uploaded file...")
            progress_bar.progress(20)
            
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extract text
            status_text.text("📄 Extracting text from PDF...")
            progress_bar.progress(40)
            
            # Process with your existing function
            status_text.text("🤖 Analyzing with AI...")
            progress_bar.progress(60)
            
            # Mock the screening process (replace with your actual function)
            results = run_candidate_screening(temp_file_path, candidate_email, job_description)
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            # Add timestamp
            results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results['candidate_name'] = uploaded_file.name.replace('.pdf', '')
            
            # Store results
            st.session_state.current_screening = results
            st.session_state.screening_results.append(results)
            
            # Clean up temp file
            os.remove(temp_file_path)
            
        st.success("Resume processed successfully!")
        
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_screening_result(results):
    st.markdown("### 📊 Screening Results")
    
    # Overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Experience Level", results['experience_level'])
    
    with col2:
        skill_match = results['skill_match']
        color = "🟢" if skill_match == "Match" else "🟡" if skill_match == "Partial Match" else "🔴"
        st.metric("Skill Match", f"{color} {skill_match}")
    
    with col3:
        st.metric("Candidate", results['candidate_name'])
    
    with col4:
        st.metric("Processed", results['timestamp'])
    
    # Detailed results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📝 Resume Preview")
        with st.expander("View Extracted Text", expanded=False):
            st.text_area("Resume Content", results['extracted_resume'], height=200, disabled=True)
        
        st.markdown("#### 🎯 Final Decision")
        decision_color = get_decision_color(results['response'])
        st.markdown(f"""
        <div style="background: {decision_color}; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4>Decision: {results['response']}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📧 Contact Information")
        st.info(f"Email: {results['candidate_email']}")
        
        # Action buttons
        if st.button("📧 Send Follow-up Email", use_container_width=True):
            send_followup_email(results)
        
        if st.button("⭐ Mark as Favorite", use_container_width=True):
            st.success("Candidate marked as favorite!")
        
        if st.button("📋 Add Notes", use_container_width=True):
            show_notes_dialog(results)

def get_decision_color(response):
    if "shortlisted" in response.lower() or "interview" in response.lower():
        return "#d4edda"
    elif "escalated" in response.lower():
        return "#fff3cd"
    else:
        return "#f8d7da"

def dashboard_interface():
    if not st.session_state.screening_results:
        st.info("No screening results available. Process some resumes first!")
        return
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    results = st.session_state.screening_results
    
    with col1:
        st.metric("Total Screened", len(results))
    
    with col2:
        matches = len([r for r in results if r['skill_match'] == 'Match'])
        st.metric("Full Matches", matches)
    
    with col3:
        interviews = len([r for r in results if 'interview' in r['response'].lower()])
        st.metric("Interviews Scheduled", interviews)
    
    with col4:
        recent = len([r for r in results if datetime.now().strftime("%Y-%m-%d") in r['timestamp']])
        st.metric("Processed Today", recent)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Skill match distribution
        skill_counts = {}
        for result in results:
            skill = result['skill_match']
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        fig_pie = px.pie(
            values=list(skill_counts.values()),
            names=list(skill_counts.keys()),
            title="Skill Match Distribution",
            color_discrete_map={
                'Match': '#28a745',
                'Partial Match': '#ffc107',
                'No Match': '#dc3545'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Experience level distribution
        exp_counts = {}
        for result in results:
            exp = result['experience_level']
            exp_counts[exp] = exp_counts.get(exp, 0) + 1
        
        fig_bar = px.bar(
            x=list(exp_counts.keys()),
            y=list(exp_counts.values()),
            title="Experience Level Distribution",
            color=list(exp_counts.values()),
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recent activity
    st.markdown("### 📅 Recent Screening Activity")
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(results)
    df = df[['timestamp', 'candidate_name', 'experience_level', 'skill_match', 'response']]
    df = df.sort_values('timestamp', ascending=False)
    
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Date & Time"),
            "candidate_name": st.column_config.TextColumn("Candidate"),
            "experience_level": st.column_config.TextColumn("Experience"),
            "skill_match": st.column_config.TextColumn("Skills"),
            "response": st.column_config.TextColumn("Decision")
        }
    )

def history_interface():
    if not st.session_state.screening_results:
        st.info("No screening history available.")
        return
    
    st.markdown("### 📋 Screening History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_experience = st.selectbox(
            "Filter by Experience",
            ["All"] + list(set([r['experience_level'] for r in st.session_state.screening_results]))
        )
    
    with col2:
        filter_skills = st.selectbox(
            "Filter by Skill Match",
            ["All"] + list(set([r['skill_match'] for r in st.session_state.screening_results]))
        )
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Timestamp", "Experience Level", "Skill Match"])
    
    # Filter results
    filtered_results = st.session_state.screening_results
    
    if filter_experience != "All":
        filtered_results = [r for r in filtered_results if r['experience_level'] == filter_experience]
    
    if filter_skills != "All":
        filtered_results = [r for r in filtered_results if r['skill_match'] == filter_skills]
    
    # Display results
    for i, result in enumerate(filtered_results):
        with st.expander(f"📄 {result['candidate_name']} - {result['timestamp']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Experience Level:** {result['experience_level']}")
                st.write(f"**Skill Match:** {result['skill_match']}")
                st.write(f"**Decision:** {result['response']}")
                st.write(f"**Email:** {result['candidate_email']}")
            
            with col2:
                if st.button(f"📧 Contact", key=f"contact_{i}"):
                    send_followup_email(result)
                
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.screening_results.remove(result)
                    st.rerun()

def display_statistics():
    results = st.session_state.screening_results
    
    if results:
        # Quick stats
        total = len(results)
        matches = len([r for r in results if r['skill_match'] == 'Match'])
        interviews = len([r for r in results if 'interview' in r['response'].lower()])
        
        st.metric("Total Processed", total)
        st.metric("Success Rate", f"{(matches/total)*100:.1f}%")
        st.metric("Interviews", interviews)

def send_followup_email(results):
    st.success(f"Follow-up email sent to {results['candidate_email']}")

def show_notes_dialog(results):
    st.text_area("Add notes for this candidate:", key=f"notes_{results['candidate_email']}")

def export_results():
    df = pd.DataFrame(st.session_state.screening_results)
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
