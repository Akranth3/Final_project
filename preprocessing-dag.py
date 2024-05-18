from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import tensorflow as tf
import os
import json

# Define image size and data augmentation function
img_size = (256, 256)

def data_augmentation(car_img, mask_img):
    if tf.random.uniform(()) > 0.5:
        car_img = tf.image.flip_left_right(car_img)
        mask_img = tf.image.flip_left_right(mask_img)
    return car_img, mask_img

def preprocessing(car_path, mask_path):
    car_img = tf.io.read_file(car_path)
    car_img = tf.image.decode_jpeg(car_img, channels=3)
    car_img = tf.image.resize(car_img, img_size)
    car_img = tf.cast(car_img, tf.float32) / 255.0
    
    mask_img = tf.io.read_file(mask_path)
    mask_img = tf.image.decode_jpeg(mask_img, channels=3)
    mask_img = tf.image.resize(mask_img, img_size)
    mask_img = mask_img[:, :, :1]
    mask_img = tf.math.sign(mask_img)
    
    car_img, mask_img = data_augmentation(car_img, mask_img)
    
    return car_img, mask_img

def input_file_sensor(base_input_dir, input_file_path):
    categories = ['Benign', 'Early', 'Pre', 'Pro']
    input_files = {}

    for category in categories:
        car_dir = os.path.join(base_input_dir, 'Original', category)
        mask_dir = os.path.join(base_input_dir, 'Segmented', category)
        
        car_paths = sorted([os.path.join(car_dir, f) for f in os.listdir(car_dir) if f.endswith('.jpg')])
        mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.jpg')])

        input_files[category] = list(zip(car_paths, mask_paths))
    
    with open(input_file_path, 'w') as f:
        json.dump(input_files, f)

def data_augmentation_task(input_file_path, preprocessed_file_path):
    with open(input_file_path, 'r') as f:
        input_files = json.load(f)
    
    preprocessed_image_paths = {}

    temp_dir = '/tmp/preprocessed_images'
    os.makedirs(temp_dir, exist_ok=True)

    for category, paths in input_files.items():
        category_paths = []
        for i, (car_path, mask_path) in enumerate(paths):
            car_img, mask_img = preprocessing(car_path, mask_path)
            
            car_output_path = os.path.join(temp_dir, f'{category}_car_{i}.jpg')
            mask_output_path = os.path.join(temp_dir, f'{category}_mask_{i}.jpg')
            
            tf.io.write_file(car_output_path, tf.io.encode_jpeg(tf.cast(car_img * 255, tf.uint8)))
            tf.io.write_file(mask_output_path, tf.io.encode_jpeg(tf.cast(mask_img * 255, tf.uint8)))

            category_paths.append((car_output_path, mask_output_path))
        
        preprocessed_image_paths[category] = category_paths
    
    with open(preprocessed_file_path, 'w') as f:
        json.dump(preprocessed_image_paths, f)

def output_preprocessed_folder(preprocessed_file_path, base_output_dir):
    with open(preprocessed_file_path, 'r') as f:
        preprocessed_image_paths = json.load(f)

    for category, paths in preprocessed_image_paths.items():
        output_car_dir = os.path.join(base_output_dir, 'Original', category)
        output_mask_dir = os.path.join(base_output_dir, 'Segmented', category)
        
        os.makedirs(output_car_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        
        for car_path, mask_path in paths:
            car_output_path = os.path.join(output_car_dir, os.path.basename(car_path))
            mask_output_path = os.path.join(output_mask_dir, os.path.basename(mask_path))
            
            os.rename(car_path, car_output_path)
            os.rename(mask_path, mask_output_path)

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'lymphoma_preprocessing_dag_pipeline',
    default_args=default_args,
    description='A DAG to preprocess and augment the ALL dataset using PySpark',
    schedule_interval=timedelta(days=1),
)

# Define tasks
start = DummyOperator(task_id='start', dag=dag)

input_file_sensor_task = PythonOperator(
    task_id='input_file_sensor',
    python_callable=input_file_sensor,
    op_kwargs={
        'base_input_dir': '/Users/vishalvignesh/codes/_cs5830/project',
        'input_file_path': '/tmp/input_files.json'
    },
    dag=dag,
)

data_augmentation_task = PythonOperator(
    task_id='data_augmentation_task',
    python_callable=data_augmentation_task,
    op_kwargs={
        'input_file_path': '/tmp/input_files.json',
        'preprocessed_file_path': '/tmp/preprocessed_files.json'
    },
    dag=dag,
)

output_preprocessed_folder_task = PythonOperator(
    task_id='output_preprocessed_folder',
    python_callable=output_preprocessed_folder,
    op_kwargs={
        'preprocessed_file_path': '/tmp/preprocessed_files.json',
        'base_output_dir': '/Users/vishalvignesh/codes/_cs5830/project/preprocessed'
    },
    dag=dag,
)

end = DummyOperator(task_id='end', dag=dag)

# Set task dependencies
start >> input_file_sensor_task >> data_augmentation_task >> output_preprocessed_folder_task >> end
