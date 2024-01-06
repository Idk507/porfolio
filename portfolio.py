import streamlit as st
import re



# Function to display home page

    
def home():
    bg_img = "./anas-alshanti-feXpdV001o4-unsplash.jpg"
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{bg_img});
        background-size: cover;
    }}  
    </style>""",
    unsafe_allow_html=True)
    st.title("Dhanushkumar R's Portfolio")
    st.image("profile_picture.jpg")
    st.header("DHANUSHKUMAR R")
    st.markdown("📞 +91 9600917002")
    st.markdown("✉️ danushidk507@gmail.com")
    st.markdown("[💻LinkedIn](https://www.linkedin.com) | [🤖GitHub](https://github.com) | [📘Medium](https://medium.com)", unsafe_allow_html=True)
    
# Function to display education page
def education():
    st.header("Education")
    st.markdown("🎓 **B.Tech Artifical Intelligence and Data Science**, St.Jospeh’s College Of Engineering, Expected 2025")
    st.markdown("🎓 CGPA 9.10")
    st.markdown("📘 Relevant Coursework: DeepLearning.ai, IBM Machine Learning")
    st.markdown("🎓 **Higher Secondary,Senthil MAtriculation Higher Secondary School")
    st.markdown("🎓 Percentage: 92%")

# Function to display skills and tools page
def skills_and_tools():
    col1, col2 = st.columns(2)
    with col1:
        st.header("Skills")
        st.markdown("""
         💻 Programming Languages: Python, Java(FOP)
         🌐 Frameworks: Django, Streamlit, Gradio, fastapi  
         📊 Visualization: Tableau
         
         🧠 ML Libraries: Numpy, Pandas, Scikit-learn  
         🤖 DL Libraries: TensorFlow, PyTorch
        """)

    with col2:
        st.header("Tools")
        st.markdown("""
         🤗 Hugging Face
         🐳 Docker 
         🖼️ LabelImg
         🚀 Streamlit, Gradio
         🚀 fastapi  
         📊 Tableau
         🧠 MNE-Python
        """)

# Function to display experience page
def experience():
    st.header("Experience")
    # Include your experience details here using st.markdown
    with st.container():
         st.header("Experience")

    st.markdown("""**Data Scientist Intern**  \
    @ Lifease Solutions""")
    
    st.markdown("""Aug 2023 - Oct 2023""")
    
    st.markdown("""
    - Working on real time and Industry Related projects.Working on voice processing and recognition
    - R&D on voice processing
    - Extracting and converting the text from json to audio
    - Audio feature extraction like mel spectrogram,fouier transformer
    - Building model for audio denoising with CNN,Bi-LSTM,AutoEncoders  
    - Using Audo-Ai_Noise Removal to denoise the data 
    - Pyttsx3 to convert audio to text for feature extraction(R&D)
    - Familiar with Libraries: Librosa,pytorch,pyqt,pyttsx3,audo ai,tensorflow
    """)

    st.markdown("""**Machine Learning Intern** \\
    @ Omdena Algebria Chapter""")

    st.markdown("""June 2023 - Present""")

    st.markdown("""
    - Goal of the Project: Develop a Comprehensive Open-Source Water Management and Forecasting System
     Create a user-friendly platform tailored to the specific needs of Algeria and Bhopal,
    integrating machine learning algorithms for precise water forecasting and efficient water resource management. 
    - Enhance Water Resource Utilization: IGmprove the sustainable use of water resources in both regions by providing accurate forecasts and real-time monitoring.
    - Capacity Building: Empower local stakeholders in Algeria and Bhopal with the knowledge and tools necessary to make well-informed decisions about water management.
    - Community Engagement: Foster collaboration among local government agencies, NGOs, and the research community to collectively address water-related challenges in both regions.
    """)    
    
    st.markdown("""**Data Science and Machine Learning Intern**  \
    @ BITA IT Training Academy """)
    
    st.markdown("""Oct 2023 - Dec 2023""")
    
    st.markdown("""
    Market Anomaly Detection is the process of identifying and analyzing unusual
    patterns in the behaviour of stocks in the stock market that cannot be explained by traditional
    market theories. One approach to Market Anomaly Detection is to analyze stock data and
    tweets. This process starts with data collection from various sources, followed by
    preprocessing,sentiment analysis, feature engineering, model development, training and
    evaluation, iterative refinement, model deployment, Analysis and Conclusions:, and
    conclusion and recommendations.
    """)

    st.markdown("""**Deep Learning Intern** \\
    @ ResoluteAI""")

    st.markdown("""June 2023 - sep 2023""")

    st.markdown("""
    - Working on YOLO model,OCR,VGG-16 models and CNN ,pytesseract ,Open CV,PIL and hands on lab on Computer vision tasks.
1)R&D on Yolo Models
2)Training and Running Yolov5 model 
3)Extracting Frames and Image annotation with labelimg,labelme
4)With Yolov8 and roboflow done object detection
5)With Yolov5 done object detection and tracking
6)With Pytorch and cv2,PIL implemented logic to detect and count the object that are passed per frame
    """)    

    st.markdown("""**Data Science Intern** \\
    @ Niograph""")

    st.markdown("""June 2023 - Present""")

    st.markdown("""
    - Working with CRM data to analyze and understand customer behavior and trends, and LLM chatbot for company website.
    """) 

# Function to display certifications page
def certifications():
    
    st.header("Certifications")
    certifications_text = """
     Generative Ai - Google Cloud Skill Batch
    * IBM AI Engineer -Batch Certification
    * IBM Deep Learning Fundamentals - IBM
    * Tensorflow Developer Certification - Deeplearning.ai
    * Convolutional Neural Networks - Deeplearning.ai
    * Time Series Analysis - Imperial College
    * Natural Language Processing - Deep Learning.ai
    * Machine Learning with Python - Cousera
    * Data Visualization with Tableau - Cousera
    * Introduction to Nosql and Database - NewtonSchool
    * Deep Learning With Pytorch - IBM
    * Matlab Fundamentals Onramp - MathWorks
    """
    for cert in certifications_text.split("*"):
        st.markdown(f"- {cert}")

# Function to display projects page
def projects():
    st.header("Projects")
    # Include your projects details here using st.markdown
    with st.container():
        st.header("Projects")

    st.markdown("""**Epilepsy Prediction with EEG Data**""")
    st.markdown("""Predicted Epilepsy with EEG dataset,worked with .edf format and using
ML and developed Deep Learning algorithm to train and predict the data .Used extracted data in .csv file which is
collection of x1 to x178 column values of brain impulses and activities and developed ’Chrononet model’ and also
used Pipeline in Ml algorithm and Deep Learning algorithm to predict the epilepsy and also Created simple UI with
streamlit for analysis""")
    
    st.markdown("""**Chatbot for Ministry of Coals**""")
    st.markdown("""LLM based chatbot- This proposal outlines the development of a 24/7 chatbot for
the mining industry, powered by transformers and offering features like 15 language translation, voice search, knowledge base/database integration, and PDF QA retrieval. This chatbot will provide stakeholders with immediate and
accurate information regarding relevant acts, rules, regulations, circulars, and land-related laws, thereby improving
efficiency, reducing workload, and increasing stakeholder satisfaction.with my team""")   
    
    st.markdown("""**Stock Market Anomally Detection for Automobile industry **""")
    st.markdown("""This research employs a multifaceted methodology for sentiment analysis and anomaly detection in the stock market, focusing on TSLA, F, NIO, and XPEV.
Sentiment scores are normalized and integrated into the dataset. It further analyzes sentiment distribution over
time and plots stock price momentum based on tweet sentiment. Anomaly detection and time series forecasting are
executed using a Bi LSTM model and an ML model, with evaluation metrics such as MAE and MSE.""")
    
    st.markdown("""**Bottle Detection and Counting**""")
    st.markdown("""- Detected and counted number of bottles using YOLO v8 algorithm
    - Annotated model using LabelImg and Roboflow for custom dataset
    - Displayed real-time detection count on video feed""")

# Function to display honors and awards page
def honors_and_awards():
    st.header("Honors & Awards")
    # Include your honors and awards details here using st.markdown
    honors = ["Winner at Hackforworld (Solve4planet)",  
              "UIPath Hackathon-1st price",
              "Sleth It Out - 2nd price",
              "SIH finalist -2023",
              "IIM Hackathon - 1st price"]
    for honor in honors:
        st.markdown(f"- {honor}")
# Function to display interests page
def interests():
    st.header("Interests")
    # Include your interests details here using st.markdown
    interests = ["Computer vision", 
                 "NLP",
                 "Generative AI",
                 "Time Series Analysis",
                 "Disease Prediction",
                 "Neuro Science",
                 "Article Writing on Medium and LinkedIn",
                 "Language Models"]

    for interest in interests:
        st.markdown(f"- {interest}")

# Sidebar navigation
pages = {
    "Home": home,
    "Education": education,
    "Skills & Tools": skills_and_tools,
    "Experience": experience,
    "Certifications": certifications,
    "Projects": projects,
    "Honors & Awards": honors_and_awards,
    "Interests": interests
}

# Streamlit app
st.set_page_config(page_title="My Portfolio", page_icon=":briefcase:")

# Sidebar navigation
selected_page = st.sidebar.radio("Navigate", list(pages.keys()))

# Render the selected page
pages[selected_page]()
