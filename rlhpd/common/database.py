# https://towardsdatascience.com/python-has-a-built-in-database-heres-how-to-use-it-47826c10648a
import sqlite3
from typing import List, Tuple 


class NoUnratedPair(Exception):
    """No unrated pair found in the database"""
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

    def insert_traj_pair(self, left_id:str, right_id:str): 
        """ Insert a pair of IDs that is yet to rate"""
        if left_id == right_id:
            raise Exception("The supplied IDs are the same")
        if left_id > right_id:
            left_id, right_id = right_id, left_id
        # sqlite3.IntegrityError occurs if they already exist
        self.c.execute(''' INSERT INTO trajectories (left_id, right_id, preference) VALUES(?, ?, ?) ''', (left_id, right_id, 0)) 
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