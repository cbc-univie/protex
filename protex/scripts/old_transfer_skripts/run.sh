for i in `seq 1 5000`
do
    echo "Run $i"
    c42a1 index:$i <npt.inp >out/npt_$i.out

    echo "Proton transfer ..."
    python3 analysis/proton_transfer.py im1h_oac_500_$i.dcd im1h_oac_500_$i.psf
    mv current_charges.str str/current_charges_$i.str
done
