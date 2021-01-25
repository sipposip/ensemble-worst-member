#! /bin/bash
export MARS_MULTITARGET_STRICT_FORMAT=1

for date in 2019-06-01 2019-07-21; do
# EFI for the requited dates only available at 00:00:00
for time in 00:00:00; do
mars<<EOF
retrieve,
class=od,
date=${date},
expver=1,
levtype=sfc,
param=167.132, # EFI
step=0-72,
stream=enfo,
time=${time},
area=62/-14/13/41,
type=efi,
grid=0.25/0.25,
target="ecmwf_ens_[date]_[time]_[param]_for_ens_worst_member.grb"
EOF
done
done

# grib conversion in mars request did not work, we have
# to do it manually
for f in *.grb; do
grib_to_netcdf ${f} -o ${f}.nc
done


