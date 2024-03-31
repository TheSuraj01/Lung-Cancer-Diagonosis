# python -m streamlit run dem.py      -> To run the app.
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64


# Custom CSS styles


def custom_css():
    style = """
<style>
   @media (prefers-color-scheme: dark) {
       /* Dark mode styles */
       h1 {
           font-family: 'Roboto', sans-serif;
           font-size: 48px;
           color: #FFFFFF;
           text-align: center;
           margin-bottom: 30px;
       }

       .uploaded-image-container {
           margin-top: 30px;
           text-align: center;
       }

       .uploaded-image {
           max-width: 100%;
           height: auto;
           border: 5px solid #336699;
           border-radius: 10px;
           box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
       }

       .prediction {
           font-family: 'Roboto', sans-serif;
           font-size: 24px;
           font-weight: bold;
           color: #00FF00;
           margin-top: 20px;
           text-align: center;
       }

       .precautions {
           font-family: 'Roboto', sans-serif;
           font-size: 18px;
           margin-top: 20px;
           text-align: left;
           padding: 20px;
           background-color: #333333;
           border: 1px solid #DDDDDD;
           border-radius: 5px;
           color: #FFFFFF;
       }

       .lung-cancer-info {
           font-family: 'Roboto', sans-serif;
           font-size: 16px;
           margin-top: 30px;
           margin-bottom: 30px;
           text-align: left;
           padding: 20px;
           background-color: #333333;
           border: 1px solid #DDDDDD;
           border-radius: 5px;
           color: #FFFFFF;
       }
   }

   @media (prefers-color-scheme: light) {
       /* Light mode styles (original styles) */
       h1 {
           font-family: 'Roboto', sans-serif;
           font-size: 48px;
           color: #333333;
           text-align: center;
           margin-bottom: 30px;

       }

       .uploaded-image-container {
           margin-top: 30px;
           text-align: center;
       }

       .uploaded-image {
           max-width: 100%;
           height: auto;
           border: 5px solid #336699;
           border-radius: 10px;
           box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
       }

       .prediction {
           font-family: 'Roboto', sans-serif;
           font-size: 24px;
           font-weight: bold;
           color: #008000;
           margin-top: 20px;
           text-align: center;
       }

       .precautions {
           font-family: 'Roboto', sans-serif;
           font-size: 18px;
           margin-top: 20px;
           text-align: left;
           padding: 20px;
           background-color: #f5f5f5;
           border: 1px solid #dddddd;
           border-radius: 5px;
       }

       .lung-cancer-info {
           font-family: 'Roboto', sans-serif;
           font-size: 16px;
           margin-top: 30px;
           margin-bottom: 30px;
           text-align: left;
           padding: 20px;
           background-color: #f5f5f5;
           border: 1px solid #dddddd;
           border-radius: 5px;
       }
   }
</style>
"""
    return style


model = tf.keras.models.load_model("ct_vgg_best_model.hdf5")

class_labels = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma",
]

# Define descriptions and precautions for each cancer type
cancer_descriptions = {
    # -----------------------------------------------------------------------------------------------------------
    "Adenocarcinoma": """
Adenocarcinoma is a type of lung cancer that originates in the cells lining the small air sacs (alveoli) located in the outer regions of the lungs. It is one of the most common types of lung cancer, particularly among non-smokers and younger individuals, although it can affect anyone regardless of smoking history. Adenocarcinoma typically grows slower than other forms of lung cancer, such as small cell lung cancer, but it can still spread aggressively if left untreated.

The development of adenocarcinoma is often associated with various risk factors, including exposure to cigarette smoke, secondhand smoke, air pollution, radon gas, asbestos, and certain genetic mutations. However, it's important to note that not all individuals with these risk factors will develop adenocarcinoma, and some individuals without these risk factors may still develop the disease.

One of the challenging aspects of adenocarcinoma is that it can be asymptomatic in its early stages, making it difficult to detect until it has advanced. As the cancer progresses, symptoms may begin to manifest, including persistent cough, shortness of breath, chest pain, wheezing, hoarseness, coughing up blood, fatigue, unexplained weight loss, and recurrent respiratory infections.

Diagnosis of adenocarcinoma typically involves a combination of imaging tests such as chest X-rays, CT scans, PET scans, and MRI scans, along with tissue biopsies to confirm the presence of cancerous cells. Once diagnosed, staging tests are performed to determine the extent of the cancer and whether it has spread to other parts of the body.

Treatment options for adenocarcinoma depend on various factors, including the stage of the cancer, the patient's overall health, and individual preferences. Common treatment modalities may include surgery to remove the tumor, chemotherapy, radiation therapy, targeted therapy, immunotherapy, or a combination of these approaches. The goal of treatment is to eliminate the cancer cells, slow down their growth, relieve symptoms, and improve the patient's quality of life.

Precautions for adenocarcinoma involve both primary prevention strategies to reduce the risk of developing the disease and secondary prevention measures for early detection and treatment. Primary prevention efforts include avoiding tobacco smoke, minimizing exposure to environmental pollutants, maintaining a healthy lifestyle with a balanced diet and regular exercise, and undergoing genetic testing for individuals with a family history of lung cancer.

Secondary prevention focuses on early detection through screening programs for individuals at high risk, such as long-term smokers or those with a family history of lung cancer. Screening tests may include low-dose CT scans, which can detect lung nodules at an early stage when they are more likely to be treatable.

Overall, adenocarcinoma of the lung presents significant challenges due to its potential for aggressive spread and late-stage detection. However, advances in research and treatment options offer hope for improved outcomes and better quality of life for individuals affected by this form of lung cancer. Early detection, combined with prompt and appropriate treatment, remains key in combating the impact of adenocarcinoma on individuals and their families.
""",
    # -----------------------------------------------------------------------------------------------------------
    "Large Cell Carcinoma": """
Large cell carcinoma, also known as large cell lung cancer, is a type of non-small cell lung cancer (NSCLC) that accounts for approximately 10-15% of all lung cancer cases. It is characterized by the presence of large, abnormal-looking cells under a microscope, which lack the distinctive features of other types of lung cancer cells.

Large cell carcinoma typically develops in the outer regions of the lungs and tends to grow and spread more rapidly than other types of NSCLC, such as adenocarcinoma or squamous cell carcinoma. Like other forms of lung cancer, large cell carcinoma is strongly associated with smoking, although non-smokers can also develop this type of cancer.

The exact cause of large cell carcinoma is not fully understood, but it is believed to result from cumulative damage to lung cells caused by exposure to carcinogens found in tobacco smoke, environmental pollutants, radon gas, asbestos, and other substances. Additionally, genetic mutations may play a role in the development of large cell carcinoma, although further research is needed to elucidate these mechanisms.

Symptoms of large cell carcinoma are similar to those of other types of lung cancer and may include persistent cough, shortness of breath, chest pain, wheezing, hoarseness, coughing up blood, fatigue, unexplained weight loss, and recurrent respiratory infections. However, because large cell carcinoma tends to grow rapidly, symptoms may become more pronounced and develop more quickly than in other forms of lung cancer.

Diagnosis of large cell carcinoma typically involves a combination of imaging tests such as chest X-rays, CT scans, PET scans, and MRI scans, along with tissue biopsies to confirm the presence of cancerous cells. Once diagnosed, staging tests are performed to determine the extent of the cancer and whether it has spread to other parts of the body.

Treatment options for large cell carcinoma depend on various factors, including the stage of the cancer, the patient's overall health, and individual preferences. Common treatment modalities may include surgery to remove the tumor, chemotherapy, radiation therapy, targeted therapy, immunotherapy, or a combination of these approaches. The goal of treatment is to eliminate the cancer cells, slow down their growth, relieve symptoms, and improve the patient's quality of life.

Precautions for large cell carcinoma involve both primary prevention strategies to reduce the risk of developing the disease and secondary prevention measures for early detection and treatment. Primary prevention efforts include avoiding tobacco smoke, minimizing exposure to environmental pollutants, maintaining a healthy lifestyle with a balanced diet and regular exercise, and undergoing genetic testing for individuals with a family history of lung cancer.

Secondary prevention focuses on early detection through screening programs for individuals at high risk, such as long-term smokers or those with a family history of lung cancer. Screening tests may include low-dose CT scans, which can detect lung nodules at an early stage when they are more likely to be treatable.

Overall, large cell carcinoma of the lung presents significant challenges due to its rapid growth and tendency to spread. However, advances in research and treatment options offer hope for improved outcomes and better quality of life for individuals affected by this form of lung cancer. Early detection, combined with prompt and appropriate treatment, remains key in combating the impact of large cell carcinoma on individuals and their families.
""",
    # -----------------------------------------------------------------------------------------------------------
    "Normal": "No cancer detected. However, it's essential to continue regular medical checkups and adopt a healthy lifestyle to prevent potential health issues.",
    # -----------------------------------------------------------------------------------------------------------
    "Squamous Cell Carcinoma": """
Squamous cell carcinoma, also known as squamous cell lung cancer, is a type of non-small cell lung cancer (NSCLC) that arises from the squamous cells lining the airways of the lungs. It accounts for approximately 25-30% of all lung cancer cases and is most commonly found in the central airways, such as the bronchi. Squamous cell carcinoma is strongly associated with cigarette smoking, although non-smokers can also develop this type of lung cancer.

The development of squamous cell carcinoma is believed to be primarily driven by exposure to carcinogens found in tobacco smoke, including polycyclic aromatic hydrocarbons and nitrosamines. These substances can cause genetic mutations and damage to the cells lining the airways, leading to the uncontrolled growth and proliferation of squamous cells.

Squamous cell carcinoma typically presents with symptoms similar to other types of lung cancer, including persistent cough, shortness of breath, chest pain, wheezing, hoarseness, coughing up blood, fatigue, unexplained weight loss, and recurrent respiratory infections. Because squamous cell carcinoma often develops in the central airways, it may obstruct airflow more quickly than other types of lung cancer, leading to more pronounced symptoms in the early stages.

Diagnosis of squamous cell carcinoma involves a combination of imaging tests such as chest X-rays, CT scans, PET scans, and MRI scans, along with tissue biopsies to confirm the presence of cancerous cells. Once diagnosed, staging tests are performed to determine the extent of the cancer and whether it has spread to other parts of the body.

Treatment options for squamous cell carcinoma depend on various factors, including the stage of the cancer, the patient's overall health, and individual preferences. Common treatment modalities may include surgery to remove the tumor, chemotherapy, radiation therapy, targeted therapy, immunotherapy, or a combination of these approaches. The goal of treatment is to eliminate the cancer cells, slow down their growth, relieve symptoms, and improve the patient's quality of life.

Precautions for squamous cell carcinoma involve both primary prevention strategies to reduce the risk of developing the disease and secondary prevention measures for early detection and treatment. Primary prevention efforts include avoiding tobacco smoke, minimizing exposure to environmental pollutants, maintaining a healthy lifestyle with a balanced diet and regular exercise, and undergoing genetic testing for individuals with a family history of lung cancer.

Secondary prevention focuses on early detection through screening programs for individuals at high risk, such as long-term smokers or those with a family history of lung cancer. Screening tests may include low-dose CT scans, which can detect lung nodules at an early stage when they are more likely to be treatable.

Overall, squamous cell carcinoma of the lung presents significant challenges due to its aggressive nature and tendency to obstruct the airways. However, advances in research and treatment options offer hope for improved outcomes and better quality of life for individuals affected by this form of lung cancer. Early detection, combined with prompt and appropriate treatment, remains key in combating the impact of squamous cell carcinoma on individuals and their families.
""",
}
# -----------------------------------------------------------------------------------------------------------

# Define what to do if cancer is predicted
cancer_advice = """

1. **Consult a Specialist:** Schedule an appointment with a qualified oncologist (cancer specialist) or pulmonologist (lung specialist) as soon as possible. They will review your medical history, order additional tests, and provide a comprehensive evaluation.

2. **Undergo Diagnostic Tests:** Your doctor may recommend additional diagnostic tests, such as a biopsy, CT scan, or PET scan, to confirm the diagnosis and determine the stage and extent of the cancer.

3. **Discuss Treatment Options:** Based on the type and stage of cancer, your doctor will recommend the most appropriate treatment plan. Treatment options may include surgery, chemotherapy, radiation therapy, targeted therapy, or a combination of these approaches.

4. **Seek Support:** Dealing with a cancer diagnosis can be emotionally and mentally challenging. Consider joining a support group or seeking counseling to help you cope with the stress and anxiety associated with the diagnosis and treatment process.

5. **Make Lifestyle Changes:** Your doctor may recommend lifestyle changes, such as quitting smoking (if applicable), improving your diet, and increasing physical activity, to support your overall health and recovery.

Remember, early detection and treatment can significantly improve the chances of successful cancer management and recovery. Do not delay seeking medical attention if cancer is suspected.
"""


# Function to preprocess the image
def preprocess_image(image):
    img = image.convert("RGB")  # Convert image to RGB format
    img = img.resize((350, 350))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    class_index = np.argmax(prediction)
    probability = prediction[class_index]
    predicted_class = class_labels[class_index]
    return predicted_class, probability


# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Lung Cancer Diagnosis",
        page_icon="🫁",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Add custom CSS styles
    st.markdown(custom_css(), unsafe_allow_html=True)

    st.title("Lung Cancer Diagnosis")

    # -------------------------------------------------------------------------------------------------------
    st.markdown(
        """
            <style>
            h1 {
            font-size: 36px;
            font-weight: bold;
            color: #336699;
            text-align: center;
            margin-bottom: 20px;
            border: 1px solid #336699;
        }
            </style>
            """,
        unsafe_allow_html=True,
    )
    # --------------------------------------------------------------------------------------------------------------

    # Add sidebar
    st.sidebar.title("About")
    st.sidebar.markdown(
        "This web application is designed to diagnose lung cancer from CT scan images using a deep learning model. The model has been trained on a dataset of CT scans to classify the input image into one of the following classes: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, or Normal (no cancer detected)."
    )
    st.sidebar.markdown(
        "To use the app, simply upload a CT scan image, and the model will provide a prediction, along with a probability score and a description of the predicted cancer type, if applicable."
    )
    st.sidebar.markdown(
        "**Note:** This application is for educational and informational purposes only and should not be used as a substitute for professional medical advice or diagnosis."
    )

    # Upload image
    uploaded_file = st.file_uploader(
        "Choose a CT scan image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        with st.spinner("Predicting..."):
            predicted_class, probability = predict(image)

        # Display prediction
        st.markdown(
            f'<p class="prediction">Predicted Class: {predicted_class}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="prediction">Probability: {probability:.2f}</p>',
            unsafe_allow_html=True,
        )

        # Display description and precautions
        if predicted_class != "Normal":
            st.markdown(
                f'<div class="precautions"><b>Description:</b> {cancer_descriptions[predicted_class]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="precautions"><b>What to do:</b> {cancer_advice}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="precautions">No precautions are necessary, but continue regular medical checkups.</div>',
                unsafe_allow_html=True,
            )

    # Add custom CSS for the "About Lung Cancer Types" section
    st.markdown(
        """
        <div class="lung-cancer-info">
            <h3>About Lung Cancer Types</h3>
            <p>There are several types of lung cancer, including:</p>
            <ul>
                <li>Adenocarcinoma</li>
                <li>Large Cell Carcinoma</li>
                <li>Squamous Cell Carcinoma</li>
                <li>Normal (No cancer detected)</li>
            </ul>
            <p>Each type may have different characteristics and treatment options. Consult a healthcare professional for more information.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def image_to_base64(image):
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    main()
