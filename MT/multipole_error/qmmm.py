from lammps import lammps

class lammps4qmmm(lammps):
    def set_qm_atoms(self, qm_indexes, ca_indexes):
        qm = [str(i+1) for i in qm_indexes]
        qm = " ".join(qm)
        qm_no_ca = [str(i+1) for i in qm_indexes if i not in ca_indexes]
        qm_no_ca = " ".join(qm_no_ca)
        self.command("group qm id " + qm)
        self.command("delete_bonds qm multi remove")
        self.command("neigh_modify exclude group qm qm")
        self.command("group qm_no_ca id " + qm_no_ca)
        self.command("set group qm_no_ca charge 0.0")
