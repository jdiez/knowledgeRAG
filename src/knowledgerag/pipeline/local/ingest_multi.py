import multiprocessing as mp

from knowledgerag import read_configuration
from knowledgerag.database.lancedb_lib.lancedb_client import LancedbDatabase
from knowledgerag.database.lancedb_lib.lancedb_common import LanceDbSettings
from knowledgerag.pipeline.local.ingestion_pipeline import set_up


class Listener:
    def __init__(self, client: LancedbDatabase, uri: str, table_name: str) -> None:
        """_summary_

        Args:
            uri (str): _description_
            tablename (str): _description_
        """
        self.db = client
        self.uri = uri
        self.table_name = table_name

    def __call__(self, q) -> None:
        """Will add the data"""
        with self.db(LanceDbSettings(uri=self.uri)) as db:
            table = db.client.open_table(self.table_name)
            while True:
                data = q.get()
                if data == "#done#":
                    break
                table.add(data=data)


configuration = read_configuration()
pipe = configuration["pipeline"]["default"]
pf = set_up(pipe=pipe)


def worker_function(filename, q):
    """
    do some work, put results in queue
    """
    res = pf(documents=filename)
    q.put(res)


if __name__ == "__main__":
    manager = mp.Manager()
    q = manager.Queue()
    file_pool = mp.Pool(1)
    file_pool.apply_async(Listener, (q,))

    pool = mp.Pool(16)
    jobs = []
    for item in range(10000):
        job = pool.apply_async(worker_function, (item, q))
        jobs.append(job)

    for job in jobs:
        job.get()

    q.put("#done#")  # all workers are done, we close the output file
    pool.close()
    pool.join()
