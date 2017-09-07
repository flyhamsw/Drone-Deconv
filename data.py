import tensorflow as tf
import sqlite3
import random
import os

def get_db_connection():
    try:
        conn = sqlite3.connect('Patch.db')
    except Exception as e:
        print(e)
    cur = conn.cursor()
    return conn, cur

def insert_ngii_dataset():
    ngii_dataset_training_dir = 'ngii_dataset_training'
    ngii_dataset_validation_dir = 'ngii_dataset_validation'
    ngii_dataset_test_dir = 'ngii_dataset_test'

    conn, cur = get_db_connection()
    dataset_training_names = os.listdir(ngii_dataset_training_dir)
    dataset_validation_names = os.listdir(ngii_dataset_validation_dir)
    dataset_test_names = os.listdir(ngii_dataset_test_dir)

    cur.execute('delete from ngii_dir;')
    cur.execute('delete from patch_dir;')

    for name in dataset_training_names:
        ngii_x_dir = '%s/%s/x.png' % (ngii_dataset_training_dir, name)
        ngii_y_dir = '%s/%s/y.png' % (ngii_dataset_training_dir, name)
        try:
            cur.execute("insert into ngii_dir values ('%s', '%s', '%s', 'training');" % (name, ngii_x_dir, ngii_y_dir))
        except Exception as e:
            print(e)
            print(name)


    for name in dataset_validation_names:
        ngii_x_dir = '%s/%s/x.png' % (ngii_dataset_validation_dir, name)
        ngii_y_dir = '%s/%s/y.png' % (ngii_dataset_validation_dir, name)
        try:
            cur.execute("insert into ngii_dir values ('%s', '%s', '%s', 'validation');" % (name, ngii_x_dir, ngii_y_dir))
        except Exception as e:
            print(e)
            print(name)


    for name in dataset_test_names:
        ngii_x_dir = '%s/%s/x.png' % (ngii_dataset_test_dir, name)
        ngii_y_dir = '%s/%s/y.png' % (ngii_dataset_test_dir, name)
        cur.execute("insert into ngii_dir values ('%s', '%s', '%s', 'test');" % (name, ngii_x_dir, ngii_y_dir))

    conn.commit()
    cur.close()
    conn.close()

def insert_drone_dataset():
    drone_dataset_dir = 'drone_dataset'

    conn, cur = get_db_connection()
    dataset_drone_names = os.listdir(drone_dataset_dir)

    cur.execute('delete from drone_dir;')

    for name in dataset_drone_names:
        drone_x_dir = '%s/%s/x.png' % (drone_dataset_dir, name)
        cur.execute("insert into drone_dir values ('%s', '%s');" % (name, drone_x_dir))

    conn.commit()
    cur.close()
    conn.close()

def get_steps_per_epoch(batch_size, interest_data):
    #select count(*) from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='training';
    conn, cur = get_db_connection()
    cur.execute("select count(*) from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='%s';" % interest_data)
    rows = cur.fetchall()
    steps = int(rows[0][0]/batch_size)
    print('%d steps / epoch' % steps)
    return steps

def get_steps_per_epoch_drone(batch_size):
    conn, cur = get_db_connection()
    cur.execute("select count(*) from drone_patch_dir;")
    rows = cur.fetchall()
    steps = int(rows[0][0]/batch_size)
    print('%d steps / epoch' % steps)
    return steps
    
def get_ngii_dir_all():
    conn, cur = get_db_connection()
    cur.execute("select * from ngii_dir;")
    ngii_dir = cur.fetchall()
    cur.close()
    conn.close()
    return ngii_dir
    
def get_drone_dir_all():
    conn, cur = get_db_connection()
    cur.execute("select * from drone_dir;")
    ngii_dir = cur.fetchall()
    cur.close()
    conn.close()
    return ngii_dir


def get_patch_all(conn, cur, purpose):
    cur.execute("select patch_dir.x_dir, patch_dir.y_dir from patch_dir inner join ngii_dir on patch_dir.name = ngii_dir.name where ngii_dir.purpose='%s';" % purpose)
    patch_dir = cur.fetchall()
    random.shuffle(patch_dir)
    patch_filenames = []
    x_patch_filenames = []
    y_patch_filenames = []
    for row in patch_dir:
        patch_filenames.append((row[0], row[1]))
        x_patch_filenames.append(row[0])
        y_patch_filenames.append(row[1])
    return patch_filenames, x_patch_filenames, y_patch_filenames
    
def get_drone_patch_all(conn, cur):
    cur.execute("select x_dir from drone_patch_dir;")
    patch_dir = cur.fetchall()
    random.shuffle(patch_dir)
    patch_filenames = []
    for row in patch_dir:
        patch_filenames.append(row[0])
    return patch_filenames        

def insert_patch(name, x_data, y_data):
    conn, cur = get_db_connection()

    if len(x_data) > len(y_data):
        num_data = len(y_data)
    else:
        num_data = len(x_data)

    for i in range(0, num_data):
        curr_dataset_name = name[i]
        x_patch_dir = x_data[i]
        y_patch_dir = y_data[i]

        cur.execute("insert into patch_dir values ('%s', '%s', '%s');" % (curr_dataset_name, x_patch_dir, y_patch_dir))

    conn.commit()
    cur.close()
    conn.close()

def insert_drone_patch(name, x_data):
    conn, cur = get_db_connection()

    for i in range(0, len(x_data)):
        curr_dataset_name = name[i]
        x_patch_dir = x_data[i]

        cur.execute("insert into drone_patch_dir values ('%s', '%s');" % (curr_dataset_name, x_patch_dir))

    conn.commit()
    cur.close()
    conn.close()
    
if __name__=='__main__':
    #insert_ngii_dataset()
    insert_drone_dataset()
