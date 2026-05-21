import numpy as np
import tensorflow as tf

def representative_data_gen():
    # Used by TFLite converter to calibrate quantization ranges
    X_train = np.load("X_train.npy").astype(np.float32)
    # Select a representative subset of 100 samples
    for i in range(100):
        sample = np.expand_dims(X_train[i], axis=0)
        yield [sample]

if __name__ == "__main__":
    # Load trained model
    model = tf.keras.models.load_model("predictive_maintenance_model.keras")
    
    # Convert to TFLite with INT8 Full Integer Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Ensure fully quantized operations
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    # Save the quantized model file
    tflite_path = "model_quant.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Quantized TFLite model saved to {tflite_path}")
    
    # Generate C source header file array for deployment
    # Equivalent to running terminal command: xxd -i model_quant.tflite > model.h
    bytes_array = [f"0x{b:02x}" for b in tflite_quant_model]
    c_array = ", ".join(bytes_array)
    
    header_content = f"""#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

const unsigned char g_model[] DATA_ALIGN_ATTRIBUTE = {{
{c_array}
}};
const unsigned int g_model_len = {len(tflite_quant_model)};

#endif // MODEL_DATA_H_
"""
    
    with open("model.h", "w") as f:
        f.write(header_content)
    print("C Header file 'model.h' successfully created.")