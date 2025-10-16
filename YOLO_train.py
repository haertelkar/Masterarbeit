from ultralytics import YOLO
model = YOLO('atoms.pt')  ## here you can also take my weights as an initial guess
path_yaml='atoms.yaml' ## here you insert the path to the yaml file you created
results = model.train(data=path_yaml, epochs=100, batch=3)
print("Training completed. Results:", results)
# Save the model after training
model.save('trained_YOLO.pt')