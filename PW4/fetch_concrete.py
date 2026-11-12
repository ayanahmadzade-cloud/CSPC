import os
from ucimlrepo import fetch_ucirepo

def main():
    concrete = fetch_ucirepo(id=165)
    
    X = concrete.data.features.copy()
    y = concrete.data.targets.iloc[:, 0].rename("strength")
    
    print("--- METADATA ---")
    print(concrete.metadata)
    print("\n--- VARIABLES ---")
    print(concrete.variables)
    
    # Sütun adlarını original sıraya uyğun dəyişirik
    X.columns = [
        "cement", "slag", "ash",
        "water", "superplastic",
        "coarse_agg", "fine_agg",
        "age"
    ]
    
    os.makedirs("data", exist_ok=True)
    df = X.assign(strength=y)
    df.to_csv("data/concrete.csv", index=False)
    print("\nSaved data/concrete.csv successfully! Shape:", df.shape)

if __name__ == "__main__":
    main()