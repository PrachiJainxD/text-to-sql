import sqlite3
import os

def convert_stringlist_to_list(str_list):
    return str_list.strip('][').split(', ')

def check_datatype(val):
    # if isinstance(val,tuple) or isinstance(val,list):
    #     if len(val)>1:
    #         return val
    #     else:
    #         return val[0]    
    # else:
    #     return val
    return val

def run_query(db_id, query):
    try:
        data_dir = f"/app/data/spider/database/{db_id}"
        db_path = os.path.join(data_dir,f'{db_id}.sqlite')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Execute an SQL query
        cursor.execute(query)

        # Fetch the results
        results = cursor.fetchall()
        res = []
        # Process the results
        for row in results:
            #res.append(check_datatype(row))
            res.append(row)
        # Close the database connection
        conn.close()
        return res
    except:
        return []