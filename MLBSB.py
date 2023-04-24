import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv("data.csv")

# Clean the data
df = df.dropna()

# Engineer features
df["points_per_game"] = df["points"] / df["games"]
df["points_per_minute"] = df["points"] / df["minutes"]
df["points_per_touchdown"] = df["points"] / df["touchdowns"]
df["points_per_reception"] = df["points"] / df["receptions"]
df["points_per_rush"] = df["points"] / df["rushes"]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df[["points_per_game", "points_per_minute", "points_per_touchdown", "points_per_reception", "points_per_rush"]], df["points"], test_size=0.25)

# Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean squared error:", np.mean((y_pred - y_test)**2))

# Deploy the model
def create_lineup(salary_cap, slate):
    # Get the list of players
    players = df[df["slate"] == slate]

    # Filter the players by salary
    players = players[players["salary"] <= salary_cap]

    # Sort the players by predicted points
    players = players.sort_values("points", ascending=False)

    # Create the lineup
    lineup = []
    for i in range(9):
        lineup.append(players.iloc[i])

    return lineup

# Create a main slate lineup
main_slate_lineup = create_lineup(50000, "main")

# Create an express slate lineup
express_slate_lineup = create_lineup(25000, "express")

# Create an after hours slate lineup
after_hours_slate_lineup = create_lineup(10000, "after_hours"
