from bayrob.database import connection


def getdistinctval(field) -> list:
    # return the distinct values for the requested field in tf
    cur = list(connection.ros_tf.tf.aggregate(
    [
        {
            "$group": {
                "_id": f"${field}"
            }
        }
    ]))
    try:
        s = sorted([a['_id'] for a in cur])
    except TypeError:
        s = sorted(list(set([str(a['_id']) for a in cur])))
    return s
