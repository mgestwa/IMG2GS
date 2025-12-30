import time
import requests
import os
from PIL import Image, ImageDraw

URL = "http://127.0.0.1:8000"

def create_dummy_image(path):
    img = Image.new('RGB', (256, 256), color='green')
    d = ImageDraw.Draw(img)
    d.rectangle([50, 50, 200, 200], fill='red')
    img.save(path)
    return path

def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                print("Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print("Waiting for server...")
    return False

def test_process():
    image_path = "api_test_image.jpg"
    create_dummy_image(image_path)
    
    print(f"Testing /process with {image_path}...")
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        resp = requests.post(f"{URL}/process", files=files)
    
    if resp.status_code == 200:
        print("Success! Received response.")
        output_file = "api_output.ply"
        with open(output_file, "wb") as f:
            f.write(resp.content)
        print(f"Saved response to {output_file} (Size: {len(resp.content)} bytes)")
    else:
        print(f"Failed! Status: {resp.status_code}")
        print(resp.text)
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)

if __name__ == "__main__":
    if wait_for_server(URL):
        test_process()
    else:
        print("Server failed to start in time.")
        exit(1)
