# Age Group Prediction (KNN)

Nebras Jafail - GA DSB-3

Predicts the age group of a face image (child, teen, young adult, adult, senior) using KNN and the UTKFace dataset, with a Streamlit app that uses the webcam.

**Live app:** https://agepredection.streamlit.app/

## How to run
1. Download UTKFace from Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new
2. Put the `UTKFace` folder next to the notebook (or change `data_dir` in the notebook).
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
| KNN (k=5) | 53% |
| KNN + GridSearchCV (k=11, distance) | 56% |

- GridSearchCV tried different `n_neighbors` and `weights` and improved the score by 3%.
- teen is the weakest group because it has the fewest images (1180).

## App
- Upload a photo or use the webcam
- The app keeps the middle of the photo, makes it gray and 32x32, then predicts the age group
- The model is saved with `joblib` because KNN is a sklearn model (Keras `model.save` only works for Keras models)

## Limitations
- KNN compares raw pixels, it does not understand face features
- Real faces do not change suddenly at 20 or 60, so border ages are hard
- Some groups have fewer images
- The dataset may not represent everyone equally
- Lighting, angle and makeup can change the prediction
