#!/bin/sed -f

# substitute all characters except a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z
# ABCDEFGHIJKLMNOPQRSTUVWXYZáíóúýéŕĺřěťšďľžčňúäô1234567890.,?!-ÁÉÍÓÚÝĹŔŠĎĽŽČŇŤŘĚ

s/&/a/g
s/[^abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúýĺŕřťšďľžčňÁÉÍÓÚÝĹŔŘŤŠĎĽŽČŇ0123456789äÄôÔ ,-?!.]/ /g
s/[ ][ ]*/ /g
s/[ ][ ]*\./\./g
