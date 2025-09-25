# Dent.ai

**AI-driven dental diagnostics platform developed by Team Peanut Butter at the Yudhya Hackathon.**

Dent.ai leverages advanced machine learning models and a robust, automated infrastructure to provide accurate, efficient, and personalized dental care recommendations. This project is containerized with **Docker** and features a full CI/CD pipeline using **GitLab CI**, with infrastructure managed by **Terraform** and server configuration handled by **Ansible**.

## Features

✅ **AI-Powered Diagnostics:** Uses machine learning to analyze dental X-rays and oral images, detecting anomalies such as caries, fractures, and infections with high accuracy.  
✅ **Comprehensive Reports:** Generates structured PDF reports with AI-driven insights, treatment recommendations, and follow-up plans.  
✅ **Integration with Clinical Systems:** Supports **FHIR** and **DICOM** for seamless interoperability with EHR and medical imaging platforms.  
✅ **User-Friendly Interface:** A web-based, intuitive UI for both dental professionals and patients.  
✅ **Continuous Learning & Improvement:** Implements an active learning approach to enhance AI performance over time.  
✅ **Automated & Scalable Deployment:** Built with Docker, Terraform, and Ansible for reliable, automated, and scalable cloud deployments.

---

## Technology Stack

* **Backend:** Python, Flask
* **AI/ML:** TensorFlow, Keras, OpenCV
* **Containerization:** Docker, Docker Compose
* **Infrastructure as Code (IaC):** Terraform
* **Configuration Management:** Ansible
* **CI/CD:** GitLab CI/CD

---

## Getting Started

You can run Dent.ai locally using Docker or deploy it to a cloud environment using the automated scripts.

### Prerequisites

* Git
* Docker and Docker Compose
* Terraform (for cloud deployment)
* Ansible (for cloud deployment)

### 1. Local Development with Docker

This is the recommended method for running the application on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/dent.ai.git](https://github.com/yourusername/dent.ai.git)
    cd dent.ai
    ```

2.  **Set Up Environment Variables**
    Create a `.env` file by copying the example and updating the values.
    ```bash
    cp .env.example .env
    # Now, edit the .env file with your keys and settings
    ```
    Your `.env` file should look like this:
    ```env
    FLASK_APP=app.py
    FLASK_ENV=development
    SECRET_KEY=your_secret_key
    FHIR_SERVER_URL=[https://r4.smarthealthit.org](https://r4.smarthealthit.org)
    GEMINI_API_KEY=your_gemini_api_key
    ```

3.  **Build and Run with Docker Compose**
    ```bash
    docker-compose up --build
    ```
    The application will be accessible at **[http://localhost:5000](http://localhost:5000)**.

---

### 2. Automated Cloud Deployment

This project includes a complete pipeline for automated deployment to a cloud provider.

* **Terraform:** The scripts in the `/terraform` directory provision the necessary cloud infrastructure (e.g., virtual machines, networking, security groups).
* **Ansible:** The playbooks in the `/ansible` directory configure the provisioned servers, install dependencies, and deploy the Dent.ai application.
* **GitLab CI/CD:** The `.gitlab-ci.yml` file defines the pipeline that automates the entire process. On a push to the `main` branch, the pipeline will:
    1.  **Build** the Docker image.
    2.  **Test** the application.
    3.  **Provision** infrastructure using Terraform.
    4.  **Configure and Deploy** the application using Ansible.

To trigger this process, simply commit and push your changes to the GitLab repository.

---

## Usage

1.  **Access the Web Interface**
    Open your browser and visit the application URL (e.g., **[http://localhost:5000](http://localhost:5000)** for local setup).

2.  **Register or Select a Patient**
    * **New Patient:** Click "New Patient" to register.
    * **Existing Patient:** Use the search bar to find and select a patient.

3.  **Upload Dental Images**
    * **X-ray Images:** Upload in the "X-ray Diagnosis" section.
    * **Oral Images:** Upload in the "Oral Disease Diagnosis" section.

4.  **Review AI-Driven Insights**
    * The AI will analyze the images and provide diagnostic insights.

5.  **Generate Reports**
    * Click "Generate Report" to create a downloadable **PDF**.

6.  **Provide Feedback**
    * Use "Submit Feedback" to help improve the AI model.

---

## Contributing

Contributions are welcome! 🚀

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-branch`).
3.  Commit your changes (`git commit -m "Add new feature"`).
4.  Push to the branch (`git push origin feature-branch`).
5.  Open a Pull Request.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Acknowledgments

🙏 Special thanks to **Yudhya Hackathon** organizers.

---

⭐ _If you like this project, don't forget to give it a star!_ ⭐
