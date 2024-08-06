import ipi_driver2 as drv
from qmmm_efv import efv_scan

atoms = drv.Atoms(efv_scan)
# socket needs to match what in nvt.xml
client = drv.SocketClient(unixsocket='scf')
client.run(atoms)
