🎓 Student Marks Predictor

A simple web app that predicts a student's marks using Machine Learning and gives study advice using a local AI assistant.

Built with Streamlit, scikit-learn, Plotly and Ollama (llama3.2).

✨ Features
📈 Predicts marks (out of 100) from hours studied, attendance, previous score and assignments
📊 Charts: Study Hours vs Marks and Attendance vs Marks
🤖 Shows model accuracy (MAE and R² score)
💬 AI assistant for study advice
📁 Project Files
File	Purpose
app.py	Main app
student_marks.csv	Training data
requirements.txt	Libraries needed
README.md	This guide
🚀 How to Run (Step by Step)
Step 1: Download the project
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
Step 2: Create a virtual environment

Windows

python -m venv venv
venv\Scripts\activate

Mac / Linux

python3 -m venv venv
source venv/bin/activate
Step 3: Install the libraries
pip install -r requirements.txt
Step 4: Install Ollama (for the AI assistant)

Download and install from https://ollama.com/download

Skip Steps 4 and 5 if you only want predictions and charts.

Step 5: Download the AI model
ollama pull llama3.2

Make sure Ollama is running (it usually starts automatically, or run ollama serve).

Step 6: Start the app
streamlit run app.py

The browser opens automatically. If not, go to http://localhost:8501

Step 7: Enter student details

Use the left sidebar to enter:

Hours Studied per Day
Attendance (%)
Previous Score
Assignments Completed
Step 8: Predict marks

Click 🚀 Predict Marks to see the predicted score.

Step 9: Ask the AI

Type a question (for example: How can I improve my marks?) and click 💬 Ask AI.

💡 Click Predict Marks first so the AI knows the predicted score.

❓ Troubleshooting
Problem	Fix
student_marks.csv was not found	Keep the CSV in the same folder as app.py
Could not connect to Ollama	Install Ollama, run ollama pull llama3.2, and make sure it is running
ModuleNotFoundError	Run pip install -r requirements.txt again
Chart has no trendline	Run pip install statsmodels
📌 Note

The model is trained on a small dataset, so predictions are only estimates. Use inputs similar to the values in student_marks.csv for best results.
