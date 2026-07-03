import os

while True:
    print("\n===== Intrusion Detection System (IDS) =====")
    print("1. Load Dataset")
    print("2. Train Gradient Boosting Model")
    print("3. Evaluate Saved Model")
    print("4. Start Real-Time IDS")
    print("5. Generate Attack Distribution Graph")
    print("6. Generate Confusion Matrix Graph")
    print("7. Generate Feature Importance Graph")
    print("8. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        os.system("python src/load_data.py")

    elif choice == "2":
        os.system("python src/gradient_boosting.py")

    elif choice == "3":
        os.system("python src/detect.py")

    elif choice == "4":
        os.system("python src/sniffer.py")

    elif choice == "5":
        os.system("python src/visualization.py")

    elif choice == "6":
        os.system("python src/confusion_matrix_plot.py")

    elif choice == "7":
        os.system("python src/feature_importance.py")

    elif choice == "8":
        print("\nThank you for using the IDS Project.")
        break

    else:
        print("\nInvalid choice. Please try again.")