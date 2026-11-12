
from analyzer import (
    load_house_data,
    analyze_house_data,
    prepare_house_data,
    train_house_price_model,
    evaluate_house_model,
    predict_house_price,
    visualize_predictions,
    export_predictions
)


def main():
    print("=" * 50)
    print("       HOUSE PRICE PREDICTION SYSTEM")
    print("=" * 50)

    try:
        # Step 1: Load dataset
        houses = load_house_data("house_data.csv")

        # Step 2: Analyze dataset
        analyze_house_data(houses)

        # Step 3: Prepare data
        X_train, X_test, y_train, y_test, preprocessor = (
            prepare_house_data(houses)
        )

        # Step 4: Train model
        model = train_house_price_model(
            X_train, y_train, preprocessor
        )

        # Step 5: Evaluate model
        predictions = evaluate_house_model(
            model, X_test, y_test
        )

        # Step 6: Predict a new house
        predict_house_price(
            model,
            area=1400,
            bedrooms=3,
            bathrooms=2,
            age=5,
            location="Urban"
        )

        # Step 7: Visualize predictions
        visualize_predictions(y_test, predictions)

        # Step 8: Export report
        export_predictions(y_test, predictions)

        print("\nProject completed successfully!")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
