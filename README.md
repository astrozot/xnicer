# XNICER algorithm

This package implements the **XNICER** algorithm (Lombardi 2018, *A&A* vol.
615, id.A174), an optimized multi-band extinction technique based on the 
extreme deconvolution of the intrinsic colors of objects observed through
a molecular cloud. **XNICER** follows a rigorous statistical approach and 
provides the full Bayesian inference of the extinction for each observed 
object. Photometric errors in both the training control field and in the
science field are properly taken into account.

The code also implements **XNICEST**, a version of **XNICER** that takes
into accounts the effects of differential extinction (i.e., unresolved
inhomogeneities in the extinction pattern) and foreground objects.

Ancillary code for the creation of registered extinction maps is also
provided.