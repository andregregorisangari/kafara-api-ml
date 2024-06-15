from fastapi import FastAPI
import pandas as pd
import numpy as np
import tensorflow as tf
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import uvicorn

app = FastAPI()

# Load CSV files
rating_df = pd.read_csv('rating_baru.csv')
place_df = pd.read_csv('place_malang.csv')

# Load the TensorFlow Lite model
model = tf.lite.Interpreter(model_path="recommender_model.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

# Helper function to encode columns
def dict_encoder(col, data):
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

# Prepare data
user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id', rating_df)
rating_df['user'] = rating_df['User_Id'].map(user_to_user_encoded)

place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id', rating_df)
rating_df['place'] = rating_df['Place_Id'].map(place_to_place_encoded)

num_users, num_places = len(user_to_user_encoded), len(place_to_place_encoded)
rating_df['Place_Ratings'] = rating_df['Place_Ratings'].values.astype(np.float32)
min_rating, max_rating = min(rating_df['Place_Ratings']), max(rating_df['Place_Ratings'])

# Prepare place_df
place_df = place_df[['Place_Id','Place_Name','Category','Rating','Price']]
place_df.columns = ['id','place_name','category','rating','price']

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int):
    # Find places visited by the user
    user_id = int(user_id)
    place_visited_by_user = rating_df[rating_df.User_Id == user_id]

    # Places not visited by the user
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys())))
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited), place_not_visited))

    outputs = []
    for sample in user_place_array:
        input_tensor = sample.reshape(1, 2).astype(np.int64)  # Ensure input tensor is of type INT64
        model.set_tensor(input_details[0]['index'], input_tensor)
        model.invoke()
        output_data = model.get_tensor(output_details[0]['index'])
        outputs.append(output_data)

    outputs = np.array(outputs)
    ratings = outputs.flatten()
    top_ratings_indices = ratings.argsort()[-7:][::-1]

    recommended_place_ids = [place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices]
    recommended_places = place_df[place_df['id'].isin(recommended_place_ids)]

    recommendations = []
    for row, i in zip(recommended_places.itertuples(), range(1, 8)):
        recommendations.append({
            "rank": i,
            "place_name": row.place_name,
            "category": row.category,
            "price": row.price,
            "rating": row.rating
        })

    # Clean the data to ensure it is JSON serializable
    for rec in recommendations:
        for key, value in rec.items():
            if isinstance(value, (np.floating, float)) and (np.isnan(value) or np.isinf(value)):
                rec[key] = None

    # Top places visited by the user
    top_place_user = (
        place_visited_by_user.sort_values(by='Place_Ratings', ascending=False)
        .head(5)
        .Place_Id.values
    )
    top_places_visited = place_df[place_df['id'].isin(top_place_user)]
    top_places = []
    for row in top_places_visited.itertuples():
        top_places.append({
            "place_name": row.place_name,
            "category": row.category
        })

    return {
        "user_id": user_id,
        "top_places_visited": top_places,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))