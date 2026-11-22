# Age Group Prediction (KNN)

Nebras Jafail - GA DSB-3

Predicts the age group of a face image (child, teen, young adult, adult, senior) using KNN and the UTKFace dataset, with a Streamlit app that uses the webcam.

## How to run
1. Download UTKFace from Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new
2. Put the `UTKFace` folder next to the notebook.
3. `pip install -r requirements.txt`
4. Run `age_prediction.ipynb` (it saves `age_knn.pkl`).
5. `streamlit run app.py`

## Age groups
| Group | Ages |
|---|---|
| child | 0-12 |
| teen | 13-19 |
| young adult | 20-35 |
| adult | 36-59 |
| senior | 60+ |

## Results
| Model | Test accuracy |
|---|---|
| KNN (k=5) | ... |
| KNN + GridSearchCV | ... |

## App
- The link: https://agepredection.streamlit.app/
- Upload a photo or use the webcam
- The app finds the face, makes it gray and 32x32, then predicts the age group

## Limitations
- KNN compares raw pixels, it does not understand face features
- Real faces do not change suddenly at 20 or 60, so border ages are hard
- Some groups have fewer images
- The dataset may not represent everyone equally
- Lighting, angle and makeup can change the prediction
