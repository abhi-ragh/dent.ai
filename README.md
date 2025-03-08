# Dent.ai

**AI-driven dental diagnostics platform developed by Team Peanut Butter at the Yudhya Hackathon.**

Dent.ai leverages advanced machine learning models and industry-standard technologies to provide accurate, efficient, and personalized dental care recommendations.

## Features

✅ **AI-Powered Diagnostics:** Uses machine learning to analyze dental X-rays and oral images, detecting anomalies such as caries, fractures, and infections with high accuracy.  
✅ **Comprehensive Reports:** Generates structured reports with AI-driven insights, treatment recommendations, and follow-up plans.  
✅ **Integration with Clinical Systems:** Supports **FHIR** and **DICOM** for seamless interoperability with EHR and medical imaging platforms.  
✅ **User-Friendly Interface:** A web-based, intuitive UI for both dental professionals and patients.  
✅ **Continuous Learning & Improvement:** Implements an active learning approach to enhance AI performance over time.

## Installation

### Prerequisites
- Python 3.8+
- Git
- pip

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/dent.ai.git
   cd dent.ai
   ```
2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables**
   Create a `.env` file in the root directory and add:
   ```env
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your_secret_key
   FHIR_SERVER_URL=https://r4.smarthealthit.org
   GEMINI_API_KEY=your_gemini_api_key
   ```
5. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be accessible at **[http://localhost:5000](http://localhost:5000)**.

## Usage

1. **Access the Web Interface**  
   Open your browser and visit **[http://localhost:5000](http://localhost:5000)**.

2. **Register or Select a Patient**  
   - **New Patient:** Click "New Patient" to register.
   - **Existing Patient:** Use the search bar to find and select a patient.

3. **Upload Dental Images**  
   - **X-ray Images:** Upload in "X-ray Diagnosis" section.
   - **Oral Images:** Upload in "Oral Disease Diagnosis" section.

4. **Review AI-Driven Insights**  
   - The AI will analyze the images and provide diagnostic insights.

5. **Generate Reports**  
   - Click "Generate Report" to create a downloadable **PDF**.

6. **Provide Feedback**  
   - Use "Submit Feedback" to help improve the AI model.


## Contributing

Contributions are welcome! 🚀

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

🙏 Special thanks to **Yudhya Hackathon** organizers.

---

⭐ _If you like this project, don't forget to give it a star!_ ⭐

