#! /bin/bash
export MARS_MULTITARGET_STRICT_FORMAT=1

for date in 2018-07-01 2018-07-02 2003-07-01 2003-07-02 2003-07-01 2006-07-02; do
for time in 12:00:00; do
mars<<EOF
retrieve,
class=od,
date=${date},
expver=1,
levtype=sfc,
number=1/to/50/by/1,
param=121.128/167.128, # t2m, and max(t2m) in last 6 hours
step=0/to/162/by/6,
stream=enfo,
time=${time},
area=62/-14/13/41,
type=pf,
grid=0.25/0.25,
target="ecmwf_ens_[date]_[time]_[param]_for_ens_worst_member.nc",
format=netcdf
EOF
done
done

# grib conversion in mars request did not work, we have
# to do it manually
for f in *.nc; do
grib_to_netcdf ${f} -o ${f}.nc
done


