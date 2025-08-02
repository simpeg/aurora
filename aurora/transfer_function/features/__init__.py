"""
This is a sandbox folder.  Probably most of what is here will wind up in mt_metadata, mtpy, or mth5.

While dealing with practical implementation of coherence sorting, it becomes clear
 that managing channel and feature nomenclature can become tricky.

We need a way to express the channel pairs that cohernce will be evaluated on.

The normal channel pairs are:
    ex-hy
    ey-hx
    ex-ref_ex
    ey-ref_ey
    hx-ref_hx
    hy-ref_hy

    where "ref_" refers to the remote reference station.

For single station processing, only the first two are relevant.
For remote reference, all six are relevant, with usage of ref_ex, ref_ey
 being __far__ less common that the magnetic channels at the remote station.

It seems useful to have the notion of a "remote reference context",
 where we can tell a processing configuration to use for example,

ey-hx
hx-ref_hx

as our coherece sorting channels, without spelling out the reference
 channel name explictly.  This is much like what happens inside
 transfer_function_helpers.process_transfer_functions, which uses the nomenclature X, Y, RR.

Within this "remote reference context", we can compute/access the appropriate coherences.

This can be revisited later in a multivariate context.

"""
