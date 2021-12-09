# https://towardsdatascience.com/python-has-a-built-in-database-heres-how-to-use-it-47826c10648a
import sqlite3
import random
from typing import List, Tuple 


class NoUnratedPair(Exception):
    """No unrated pair found in the database"""
    pass
class NoRatedPair(Exception):
    """No rated pair found in the database"""
    pass

class AnnotationBuffer:
    def __init__(self, db_path) -> None:
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.create_table()

    def table_exists(self): 
        self.c.execute('''SELECT count(name) FROM sqlite_master WHERE TYPE = 'table' AND name = 'trajectories' ''') 
        if self.c.fetchone()[0] == 1: 
            return True 
        return False

    def create_table(self):
        if not self.table_exists(): 
            self.c.execute(''' 
                CREATE TABLE trajectories( 
                    left_id TEXT, 
                    right_id TEXT, 
                    preference INTEGER,
                    unique (left_id, right_id)
                ) ''')

    def insert_many_traj_tuples(self, traj_tuples):
        """
        traj_tuples is a list of tuples (left_id, right_id, pref)
        This is more efficient than its counterpart `insert_traj_pair`
        """
        # Data validation
        for i, (left_id, right_id, pref) in enumerate(traj_tuples):
            if left_id == right_id:
                raise Exception("The supplied IDs are the same")
            if left_id > right_id:
                traj_tuples[i][0], traj_tuples[i][1] = right_id, left_id
                if pref == 1:
                    traj_tuples[i][2] = 2
                elif pref == 2:
                    traj_tuples[i][2] = 1
        self.c.executemany(''' INSERT OR REPLACE INTO trajectories (left_id, right_id, preference) VALUES(?, ?, ?) ''', traj_tuples)
        self.conn.commit()

    def insert_traj_pair(self, left_id:str, right_id:str, pref=0): 
        """ Insert a pair of IDs, optionally with a rating"""
        if left_id == right_id:
            raise Exception("The supplied IDs are the same")
        if left_id > right_id:
            left_id, right_id = right_id, left_id
            if pref == 1:
                pref = 2
            elif pref == 2:
                pref = 1
        # sqlite3.IntegrityError occurs if they already exist
        self.c.execute(''' INSERT INTO trajectories (left_id, right_id, preference) VALUES(?, ?, ?) ''', (left_id, right_id, pref)) 
        self.conn.commit()

    def rate_traj_pair(self, left_id, right_id, preference): 
        """ Rate a pair of IDs: 1 for left, 2 for right, 3 for equally good, 4 for undecided"""
        if left_id == right_id:
            raise Exception("The supplied IDs are the same")
        if left_id > right_id:
            left_id, right_id = right_id, left_id

        if self.get_rating_of_pair(left_id, right_id) != 0:
            raise Exception("This pair was already rated")
        stmt = '''UPDATE trajectories SET 'preference' = ? WHERE left_id = ? AND right_id = ? '''
        self.c.execute(stmt,(preference, left_id, right_id))
        self.conn.commit()

    def get_all_rated_pairs_with_labels(self) -> List[Tuple[str]]:
        self.c.execute('''SELECT * FROM trajectories WHERE preference = 1 OR preference = 2 OR preference = 3''') 
        rows = self.c.fetchall()
        rated_pairs = [(line[0],line[1]) for line in rows]
        labels = [(line[2]) for line in rows]
        return rated_pairs, labels

    def label_to_judgement(self, label):
        """
        1 : (1,0)
        2 : (0,1)
        3 : (0.5,0.5)
        """
        if label == 1:
            return (1,0)
        elif label == 2:
            return (0,1)
        elif label == 3:
            return (0.5,0.5)
        else:
            raise ValueError(f"Invalid label {label}")

    def get_all_unrated_pairs(self) -> List[Tuple[str]]:
        self.c.execute('''SELECT * FROM trajectories WHERE preference = 0''') 
        unrated_pairs = self.c.fetchall()
        return [(line[0],line[1]) for line in unrated_pairs]

    def get_number_of_unrated_pairs(self) -> int:
        self.c.execute('''SELECT * FROM trajectories WHERE preference = 0''') 
        unrated_pairs = self.c.fetchall()
        return len(unrated_pairs)

    def get_one_unrated_pair(self) -> Tuple[str]:
        self.c.execute('''SELECT * FROM trajectories WHERE preference = 0 LIMIT 1''') 
        unrated_pairs = self.c.fetchall()
        if len(unrated_pairs)==0:
            raise NoUnratedPair
        return (unrated_pairs[0][0], unrated_pairs[0][1])

    def get_random_rated_tuple(self) -> Tuple[str]:
        self.c.execute('''SELECT * FROM trajectories WHERE preference = 1 OR preference = 2''') 
        rated_tuples = self.c.fetchall()
        if len(rated_tuples)==0:
            raise NoRatedPair
        rated_tuple = random.choice(rated_tuples)
        return rated_tuple

    def get_rating_of_pair(self, left_id, right_id) -> int:
        """0 for unrated, 1 for left, 2 for right, 3 for equally good, 4 for undecided"""
        if left_id == right_id:
            raise Exception("The supplied IDs are the same")
        if left_id > right_id:
            left_id, right_id = right_id, left_id

        self.c.execute('''SELECT * FROM trajectories WHERE left_id = ? AND right_id = ?''',(left_id,right_id))
        traj_pair = self.c.fetchall()
        if  len(traj_pair)==0:
            raise Exception("Pair not found in database")
        if  len(traj_pair)>1:
            raise Exception("Pair found twice in database - this is due to a bugnot supposed to happen.") 

        return traj_pair[0][2]


    ## Utilities only for testing

    def delete_pair(self, left_id, right_id):
        """ Delete a pair of IDs - this is for testing, no actual use case"""
        if left_id == right_id:
            raise Exception("The supplied IDs are the same")
        if left_id > right_id:
            left_id, right_id = right_id, left_id

        self.c.execute('''DELETE FROM trajectories WHERE left_id = ? AND right_id = ?''',(left_id,right_id)) 
        self.conn.commit()

    def return_all_data(self):
        self.c.execute('''SELECT * FROM trajectories''') 
        data = []
        for row in self.c.fetchall(): 
            data.append(row) 
        return data

    def return_all_ids(self):
        self.c.execute('''SELECT DISTINCT LEFT_ID FROM trajectories UNION SELECT DISTINCT RIGHT_ID FROM trajectories''') 
        data = []
        for row in self.c.fetchall(): 
            data.append(row) 
        return data


if __name__ == '__main__':
    pass

    # import numpy as np
    # print(np.load(f"trajectories/285223_traj_0_smpl_0.npy"))

    # Uncomment this to create test database for videos
    # db = AnnotationBuffer()
    # db.create_table()
    # db.insert_traj_pair("1000", "1001")
    # db.insert_traj_pair("1000", "1003")
    # db.insert_traj_pair("1001", "1002")
    # print(db.return_all_ids())
    # print(db.return_all_data())