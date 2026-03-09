"""
Convert your collected data (30, 1662) to pretrained model format (30, 543, 3)

Your current data structure (from data_collection.py):
- Pose: 33 landmarks × 4 coords (x, y, z, visibility) = 132 values [0:132]
- Face: 468 landmarks × 3 coords (x, y, z) = 1404 values [132:1536]
- Left Hand: 21 landmarks × 3 coords (x, y, z) = 63 values [1536:1599]
- Right Hand: 21 landmarks × 3 coords (x, y, z) = 63 values [1599:1662]

Pretrained model expects:
- Shape: (30, 543, 3)
- Order: Face -> Left Hand -> Pose -> Right Hand
- All coordinates as (x, y, z) only (no visibility)
"""

import numpy as np
import os


def convert_to_model_shape(user_sequence):
    """
    Convert your data format to pretrained model format.
    
    Args:
        user_sequence: numpy array of shape (30, 1662)
                      Your collected sequence data
    
    Returns:
        numpy array of shape (30, 543, 3)
        Ready for pretrained model input
    """
    # Verify input shape
    if user_sequence.shape != (30, 1662):
        raise ValueError(f"Expected shape (30, 1662), got {user_sequence.shape}")
    
    frames = []
    
    for frame in user_sequence:
        # Extract each component from YOUR data structure
        # Based on your extract_keypoints function order: [pose, face, lh, rh]
        
        # 1. POSE: indices 0-131 (132 values = 33 landmarks × 4 coords)
        #    Keep only x, y, z (drop visibility which is every 4th value)
        pose_raw = frame[0:132].reshape(33, 4)  # Shape: (33, 4)
        pose = pose_raw[:, :3]                   # Drop visibility -> (33, 3)
        
        # 2. FACE: indices 132-1535 (1404 values = 468 landmarks × 3 coords)
        #    Already in (x, y, z) format
        face = frame[132:1536].reshape(468, 3)   # Shape: (468, 3)
        
        # 3. LEFT HAND: indices 1536-1598 (63 values = 21 landmarks × 3 coords)
        #    Already in (x, y, z) format
        lh = frame[1536:1599].reshape(21, 3)     # Shape: (21, 3)
        
        # 4. RIGHT HAND: indices 1599-1661 (63 values = 21 landmarks × 3 coords)
        #    Already in (x, y, z) format
        rh = frame[1599:1662].reshape(21, 3)     # Shape: (21, 3)
        
        # Concatenate in the order expected by pretrained model:
        # Face -> Left Hand -> Pose -> Right Hand
        model_frame = np.concatenate([face, lh, pose, rh])  # Shape: (543, 3)
        
        # Verify the frame shape
        assert model_frame.shape == (543, 3), f"Frame shape mismatch: {model_frame.shape}"
        
        frames.append(model_frame)
    
    result = np.array(frames)  # Shape: (30, 543, 3)
    
    # Final verification
    assert result.shape == (30, 543, 3), f"Output shape mismatch: {result.shape}"
    
    return result


def convert_sequence_file(sequence_folder_path):
    """
    Load a sequence from disk and convert it to model format.
    
    Args:
        sequence_folder_path: Path to sequence folder containing 0.npy to 29.npy
                             Example: "MP_Data/hello/user@email.com_0"
    
    Returns:
        numpy array of shape (30, 543, 3)
    """
    # Load all 30 frames
    frames = []
    for i in range(30):
        frame_path = os.path.join(sequence_folder_path, f"{i}.npy")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame {i} not found: {frame_path}")
        
        keypoints = np.load(frame_path)  # Shape: (1662,)
        frames.append(keypoints)
    
    # Stack into sequence
    user_sequence = np.array(frames)  # Shape: (30, 1662)
    
    # Convert to model format
    model_input = convert_to_model_shape(user_sequence)
    
    return model_input


def verify_conversion():
    """
    Verify the conversion is correct by checking dimensions.
    """
    print("=" * 70)
    print("CONVERSION VERIFICATION")
    print("=" * 70)
    
    # Create dummy data matching your structure
    dummy_sequence = np.random.rand(30, 1662)
    
    print(f"\n✅ Input shape: {dummy_sequence.shape}")
    print(f"   - 30 frames")
    print(f"   - 1662 keypoints per frame")
    
    # Convert
    converted = convert_to_model_shape(dummy_sequence)
    
    print(f"\n✅ Output shape: {converted.shape}")
    print(f"   - 30 frames")
    print(f"   - 543 landmarks")
    print(f"   - 3 coordinates (x, y, z)")
    
    # Verify landmark counts
    print(f"\n📊 Landmark breakdown:")
    print(f"   - Face: 468 landmarks")
    print(f"   - Left Hand: 21 landmarks")
    print(f"   - Pose: 33 landmarks")
    print(f"   - Right Hand: 21 landmarks")
    print(f"   - Total: {468 + 21 + 33 + 21} = 543 landmarks ✓")
    
    # Check one frame structure
    frame_0 = converted[0]
    print(f"\n🔍 Frame 0 structure:")
    print(f"   - Face: indices [0:468] = {frame_0[0:468].shape}")
    print(f"   - Left Hand: indices [468:489] = {frame_0[468:489].shape}")
    print(f"   - Pose: indices [489:522] = {frame_0[489:522].shape}")
    print(f"   - Right Hand: indices [522:543] = {frame_0[522:543].shape}")
    
    print(f"\n✅ Conversion successful!")
    print(f"   Your data is now compatible with the pretrained model!")
    
    return converted


def batch_convert_sign_data(sign_name, data_path="MP_Data", output_path="MP_Data_Converted"):
    """
    Convert all sequences for a specific sign.
    
    Args:
        sign_name: Name of the sign (e.g., "hello")
        data_path: Path to your original data
        output_path: Path to save converted data
    
    Returns:
        Number of sequences converted
    """
    sign_path = os.path.join(data_path, sign_name)
    
    if not os.path.exists(sign_path):
        raise FileNotFoundError(f"Sign folder not found: {sign_path}")
    
    # Create output directory
    output_sign_path = os.path.join(output_path, sign_name)
    os.makedirs(output_sign_path, exist_ok=True)
    
    # Find all sequence folders
    sequences = [d for d in os.listdir(sign_path) 
                 if os.path.isdir(os.path.join(sign_path, d))]
    
    converted_count = 0
    
    for seq_folder in sequences:
        seq_path = os.path.join(sign_path, seq_folder)
        
        try:
            # Convert sequence
            converted_data = convert_sequence_file(seq_path)
            
            # Save converted data
            output_file = os.path.join(output_sign_path, f"{seq_folder}.npy")
            np.save(output_file, converted_data)
            
            converted_count += 1
            print(f"✅ Converted: {seq_folder}")
            
        except Exception as e:
            print(f"❌ Failed to convert {seq_folder}: {e}")
    
    print(f"\n✅ Converted {converted_count}/{len(sequences)} sequences for '{sign_name}'")
    return converted_count


# Example usage functions
def example_single_sequence():
    """Example: Convert a single sequence."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Convert Single Sequence")
    print("=" * 70)
    
    # Path to your sequence
    sequence_path = "MP_Data/hello/user@email.com_0"
    
    # Convert
    model_input = convert_sequence_file(sequence_path)
    
    print(f"\n✅ Converted sequence shape: {model_input.shape}")
    print(f"   Ready to feed into pretrained model!")
    
    return model_input


def example_batch_convert():
    """Example: Convert all sequences for a sign."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Convert All Sequences")
    print("=" * 70)
    
    # Convert all sequences for "hello"
    batch_convert_sign_data("hello")


if __name__ == "__main__":
    # Run verification
    verify_conversion()
    
    # Uncomment to test with real data:
    # example_single_sequence()
    # example_batch_convert()

