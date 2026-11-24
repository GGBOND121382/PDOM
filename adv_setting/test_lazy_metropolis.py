"""Run directly: python -m adv_setting.test_lazy_metropolis"""
import math
import numpy as np
from optimization_utils.BanditLogisticRegression import _build_network_topology, _build_accelerated_gossip_matrix
from optimization_utils.adv_setting_utils import sigma2_of_gossip

def main():
    for topo,n,want in [('grid',9,24.2105325169),('grid',36,62.4247291432),('cycle',36,149.2727511252),('complete',36,15.0226280193)]:
        adj,old,*_= _build_network_topology(topo,n)
        p=_build_accelerated_gossip_matrix(topo,n)
        assert np.allclose(p,p.T) and np.allclose(p.sum(0),1) and np.allclose(p.sum(1),1)
        assert np.all(p>=0) and np.array_equal(p>0,adj|np.eye(n,dtype=bool))
        eig=np.linalg.eigvalsh(p)
        assert eig[0]>0 and eig[-2]<1
        sigma=sigma2_of_gossip(p)
        ct=math.sqrt(2)*math.log(math.sqrt(14*n))/((math.sqrt(2)-1)*math.sqrt(1-sigma))
        assert math.isclose(ct,want,rel_tol=1e-9)
        assert np.array_equal(old,_build_network_topology(topo,n)[1])
        print(topo,n,eig[0],sigma,ct)
if __name__=='__main__':main()
