J/A+A/682/A29       Detection of the Yarkovsky effect on NEAs   (Fenucci+, 2024)
================================================================================
An automated procedure for the detection of the Yarkovsky effect
and results from the ESA NEO Coordination Centre.
    Fenucci M., Micheli M., Gianotto F., Faggioli L., Oliviero D., Porru A.,
    Rudawska R., Cano J.L., Conversi L., Moissl R.
    <Astron. Astrophys. 682, A29 (2024)>
    =2024A&A...682A..29F        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Solar system ; Minor planets
Keywords: methods: statistical - minor planets, asteroids: general

Abstract:
    The measurement of the Yarkovsky effect on near-Earth asteroids (NEAs)
    is common practice in orbit determination today, and the number of
    detections will increase with the developments of new and more
    accurate telescopic surveys. However, the process of finding new
    detections and identifying spurious ones is not yet automated, and it
    often relies on personal judgment.

    We aim to introduce a more automated procedure that can search for NEA
    candidates to measure the Yarkovsky effect, and that can identify
    spurious detections.

    The expected semi-major axis drift on an NEA caused by the Yarkovsky
    effect was computed with a Monte Carlo method on a statistical model
    of the physical parameters of the asteroid that relies on the most
    recent NEA population models and data. The expected drift was used to
    select candidates in which the Yarkovsky effect might be detected,
    according to the current knowledge of their orbit and the length of
    their observational arc. Then, a nongravitational acceleration along
    the transverse direction was estimated through orbit determination for
    each candidate. If the detected acceleration was statistically
    significant, we performed a statistical test to determine whether it
    was compatible with the Yarkovsky effect model. Finally, we determined
    the dependence on an isolated tracklet.

    Among the known NEAs, our procedure automatically found 348 detections
    of the Yarkovsky effect that were accepted. The results are overall
    compatible with the predicted trend with the the inverse of the
    diameter, and the procedure appears to be efficient in identifying and
    rejecting spurious detections. This algorithm is now adopted by the
    ESA NEO Coordination Centre to periodically update the catalogue of
    NEAs with a measurable Yarkovsky effect, and the results are
    automatically posted on the web portal.

Description:
    The Yarkovsky effect determinations can be used to extrapolate the
    ratio of retrograde-to-prograde rotators (R/P), because negative
    (positive) semi-major axis drift values are associated with retrograde
    (prograde) rotators. To this purpose, we randomly generate 10000
    values of A2, for each positive detections. In this process, we
    assumed A2 to be a Gaussian random variable, with mean value equal to
    the nominal value and standard deviation equal to the 1-{sigma}
    uncertainty of the detection.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
tableb1.dat      185      463   Determinations of A2 with S/N>3
tableb2.dat      119      350   Comparisons with JPL data
tableb3.dat      120       64   Comparisons with JPL data not accepted or
                                 not found by NEOCC
--------------------------------------------------------------------------------

See also:
     B/astorb : Orbits of Minor Planets (Bowell+, 2014-)

Byte-by-byte Description of file: tableb1.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label      Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---       Asteroid   Name of the asteroid
  12- 17  F6.3  mag       H          Absolute magnitude
  19- 26  F8.6  ---       RMS        RMS of normalized residuals, 7D OD
  28- 35  F8.6  ---       RMS6D      RMS of normalized residuals, 6D OD
  37- 48  E12.6 au/d2     A2         Transversal acceleration component
  50- 60  E11.6 au/d2   e_A2         Error in A2
  62- 73  E12.6 au/Myr    da/dt      Semi-major axis drift
  75- 85  E11.6 au/Myr  e_da/dt      Error in da/dt
  87- 97  E11.9 au/Myr    max(da/dt) Maximum da/dt from Monte Carlo model
  99-107  F9.5  ---       SNR        Signal-to-noise of A2 detection
 109-112  A4    ---       FAccept    [1 Rej.] Flag for acceptance of the
                                      detection
 114-118  I5    ---       NOptObs    Number of optical observations
 120-121  I2    ---       NRejOpt    Number of rejected optical observations
                                      in 7D OD
 123-124  I2    ---       NRej6D     Number of rejected optical observations
                                      in 6D OD
 126-127  I2    ---       NRadObs    Number of radar observations
     129  I1    ---       NRejRad    Number of rejected radar obs in 7D OD
 131-132  I2    ---       NRejRad6D  Number of rejected radar obs in 6D OD
 134-137  I4    ---       NOptOld    Number of old observations
 139-147  F9.3  m         Dlow       15-th percentile of diameter
 149-157  F9.3  m         Dmed       50-th percentile of diameter
 158-166  F9.3  m         Dhigh      85-th percentile of diameter
     168  I1    ---       ModFlag    [0/1] Flag for model used in Monte Carlo
 170-176  F7.3  h         Prot       ?=-1 Rotation period of the asteroid
     178  A1    ---       Tax        Taxonomic complex of the asteroid
 180-185  F6.3  yr        deltat     Length of observational arc
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tableb2.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---       Asteroid  Name of the asteroid
  12- 23  E12.5 au/d2     A2        Transversal acceleration from NEOCC
  25- 35  E11.5 au/d2   e_A2        Error in A2 from NEOCC
  37- 45  F9.5  ---       SNR       Signal-to-noise of A2 from NEOCC
  47- 59  E13.5 au/d2     A2j       ?=- Transversal acceleration from JPL
  61- 69  E9.5  au/d2   e_A2j       ?=- Error in A2 from JPL
  71- 79  F9.5  ---       SNRj      ?=- Signal-to-noise of A2 from JPL
      81  I1    ---       Model     [1/3]?=- Dynamical model of JPL
  83- 95 F13.10 ---       Err1      ?=- Relative error of JPL determination
  97-107 F11.9  ---       Err2      ?=- Relative error of NEOCC determination
 109-119 F11.9  ---       chiA2     ?=- Chi value of determination
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tableb3.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label        Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---      Asteroid     Name of the asteroid
  13- 25  E13.5 au/Myr   da/dt(NEOCC) ?=- Semi-major axis drift from NEOCC
  28- 46  E19.5 au/Myr e_da/dt(NEOCC) ?=- Error of NEOCC determination
  49- 58  F10.7 ---      SNR(NEOCC)   ?=- Signal-to-noise of NEOCC determination
  63- 73  F11.9 ---      YM           ?=- Maximum da/dt from Monte Carlo model
  76- 87  E12.9 au/Myr   da/dt(JPL)   Semi-major axis drift from JPL
  89- 99  E11.5 au/Myr e_da/dt(JPL)   Error of JPL determination
 103-111  F9.6  ---      SNR(JPL)     Signal-to-noise of JPL determination
     120  I1    ---      Model        [1/3] Dynamical model used by JPL
--------------------------------------------------------------------------------

Acknowledgements:
    Marco Fenucci, marco.fenucci(at)ext.esa.int

================================================================================
(End)             Marco Fenucci [RM, Italy], Patricia Vannier [CDS]  17-Nov-2023