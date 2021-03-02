/* 
 * purpose:      model the event of infection with avian flu virus 
 *               substrain H7N9 known to also affect humans; infection 
 *               will be said to occur when a random number from the 
 *               interval [0,1] falls into a certain range representing
 *               the fraction of lung/bronchi/bronchiole cells within all
 *               types of cells forming the body; in case of infection, the 
 *               progress of the virus is modeled in terms of numbers of 
 *               infected cells growing over time;
 *               n.b.  here the immune system is taken into account via a
 *                     term counteracting the increase in nmb_infected_cells[]
 *               n.b.2 again, the generalized variant is considered, 
 *                     allowing all kinds of parallel infections;       
 *               n.b.3 here the time of action of immune system components
 *                     is modeled every hour while viral distribution updates
 *                     are taken into account only every 14 hours following
 *                     the known H7N9 transmission period
 *               n.b.4 here the core part of the simulation is run on the 
 *                     GPU with basic usage of managed unified memory;
 *               n.b.5 the standard math lib needs to be replaced with the
 *                     CUDA-internal math library, which is automatically
 *                     included when calling nvcc, so just dropping -lm
 *                     on the link line will facilitate this replacement
 * result:       H7N9 infection initially starts with exponential growth   
 *               and becomes downregulated by the modulating function of
 *               the immune system; much more frequent counteraction 
 *               does render the growth-pattern now fan-shaped; however
 *               individual cells won't be synchronized that perfectly, so 
 *               the times of real updates will vary to some extent probably 
 *               leaving behind an overall gaussian-like average impression
 *               of viral invasion and defense;
 *               GPU-ported results are identical to the orignial CPU-only 
 *               results and the math lib is perfectly substituted from
 *               CUDA internal libs;
 * compilation:  nvcc ./h7n9_v4.cu 
 * usage:        ./a.out  >  ./h7n9_v4.gpu.0.dat 
 *               tail -499 ./h7n9_v4.gpu.0.dat  >  ./h7n9_v4.gpu.1.dat
 *               gnuplot  "./h7n9_v4.gpu.gnuplot"
 *               gs ./h7n9_v4.gpu.eps                       
 */


/*
 * doi: 10.3109/03014460.2013.807878
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define NMB_CELL_TYPES 54
#define NMB_INITIAL_INFECTION_ATTEMPTS 100
#define H7N9_TISSUE_TARGET 25     
#define H7N9_REPRODUCTION_NMB 22           
#define H7N9_TRANSMISSION_TIME 14     
#define H7N9_TISSUE_SURVIVAL_PERCENTAGE 35     
#define TIME_STEP_IN_HOURS 1
#define NMB_TIME_STEPS_2STUDY 500
#define HOURS_2ACTIVATE_IMMUNE_SYSTEM 24


/*
 * prototype declarations to avoid including additional local header files
 */
void initialize(float *, char **);
void random_infection(float, float *, float *);
void monitor_infection_progress(float *, float *, float *);







/* 
 * GPU kernel 
 */
__global__ void update_nmb_infected_cells(int time,
                                          float *sick_cell_type_count,
                                          float *nmb_infected_cells)
{
    int i;

    i = threadIdx.x;
    if (i == H7N9_TISSUE_TARGET) {
       if ((time % H7N9_TRANSMISSION_TIME) == 0) {

         /*
          * at this point, all currently infected cells will die and 
          * release their load of new mature H7N9 virus particles;
          */
          sick_cell_type_count[i] -= nmb_infected_cells[i];
          nmb_infected_cells[i] *= (float) H7N9_REPRODUCTION_NMB;

       }

      /*
       * consider counteraction of the immune system after an
       * initial lag time for alerting; effective action will be 
       * down-scaling of nmb_infected_cells[] 
       */
       if (time >= HOURS_2ACTIVATE_IMMUNE_SYSTEM) {
          nmb_infected_cells[i] *= exp(-0.000010969388 * time * time);
       }
    } else {

      /* 
       * just dummy actions of potential interest for 
       * cross-infections;
       */
       sick_cell_type_count[i] += 0.0;
       nmb_infected_cells[i] *= 1.0;
    }
}







/* 
 * host main  
 */
int main()
{
    int i, got_hit_by_h7n9;
    float healthy_tot_nmb_cells, *healthy_cell_type_count, *initial_infection; 
    float *sick_cell_type_count, *nmb_infected_cells;
    char **healthy_cell_type_tissue;  
    time_t t;



   /*
    * initialize reference data with values from the literature
    */
    srand((unsigned) time(&t));
    healthy_cell_type_count = (float *) malloc(NMB_CELL_TYPES * sizeof(float)); 
    healthy_cell_type_tissue = (char **) malloc(NMB_CELL_TYPES * sizeof(char *)); 
    for (i=0; i<NMB_CELL_TYPES; i++) {
        healthy_cell_type_tissue[i] = (char *) malloc(300 * sizeof(char)); 
    }
    initialize(healthy_cell_type_count, healthy_cell_type_tissue);

   /*
    * let us also allocate another array, sick_cell_type_count[],
    * for modelling an ongoing infection; initially this array will be 
    * quasi-identical to healthy_cell_type_count[], except that for a    
    * particular tissue the count (number of still healthy cells) will be 
    * lower than its corresponding counterpart in healthy_cell_type_count[] 
    * and with progress of the disease this gap will become larger and larger; 
    * at the same instance we may also introduce another array, 
    * nmb_infected_cells[], that keeps track of the current number of cells 
    * actually infected (hence still living) for each of the known tissues; 
    * this way we can take into account that whenever new cells get infected 
    * others must have died, hence distinguish between cells carrying on
    * the spread of the virus and cells already extinguished and thus just
    * missing in the overall count of functioning cells;
    * n.b. since these arrays will be used on the GPU we shall make use
    *      of managed unified memory via cudaMallocManaged()
    */
    cudaMallocManaged(&sick_cell_type_count, NMB_CELL_TYPES * sizeof(float));
    cudaMallocManaged(&nmb_infected_cells, NMB_CELL_TYPES * sizeof(float));

   /*
    * compute total number of cells including all various tissues
    */
    healthy_tot_nmb_cells = (float) 0;
    for (i=0; i<NMB_CELL_TYPES; i++) {
        printf("%6d%12.2e%*c%-s\n", i, healthy_cell_type_count[i], 
                                    5, ' ', healthy_cell_type_tissue[i]);
        healthy_tot_nmb_cells += healthy_cell_type_count[i];
    }
    printf("*** healthy: sum of all cells %12.2e ***\n", healthy_tot_nmb_cells);

   /*
    * fill a vector with all 0 except one position where a -1 shall
    * reflect infection of that particular tissue
    */
    initial_infection = (float *) malloc(NMB_CELL_TYPES * sizeof(float)); 
    random_infection(healthy_tot_nmb_cells, 
                     healthy_cell_type_count, initial_infection);
    //printf("*** infection vector ***\n");
    //for (i=0; i<NMB_CELL_TYPES; i++) {
    //    printf("%6d%6.0f\n", i, initial_infection[i]); 
    //}

   /*
    * give it a couple of attempts of initial infection, in particular 
    * NMB_INITIAL_INFECTION_ATTEMPTS times, and see whether any of them 
    * will affect lung/bronchi tissue as this is the entry point of H7N9;
    * in case of really hitting lung/bronchi, variable got_hit_by_h7n9 will 
    * be set to 1, otherwise got_hit_by_h7n9 = 0 shall signal no infection 
    * has taken place;
    */
    got_hit_by_h7n9 = 0;
    for (i=0; i<NMB_INITIAL_INFECTION_ATTEMPTS; i++) {
        random_infection(healthy_tot_nmb_cells,
                         healthy_cell_type_count, initial_infection);
        if (( initial_infection[21] + initial_infection[22] 
              + initial_infection[23] + initial_infection[24]
              + initial_infection[25] + initial_infection[26]
              + initial_infection[27] + initial_infection[28]
              + initial_infection[29] + initial_infection[30]
              + initial_infection[31]) < 0 ) {
           got_hit_by_h7n9 = 1;
        }
    }
    if (got_hit_by_h7n9 == 1) {
       printf("*** infected with H7N9 ***\n");
    }
    else {
       printf("*** not infected with H7N9 ***\n");
    }

   /*
    * depending on whether or not we have emerged as "infected" let us 
    * enter a special routine to model progress of the disease;
    */
    if (got_hit_by_h7n9 == 1) {
       monitor_infection_progress(healthy_cell_type_count, 
		                  sick_cell_type_count, nmb_infected_cells);
    }





   /*
    * and don't forget to free all allocated memory
    * n.b. now we have two instances of cudaMallocManaged() types !
    */
    free(initial_infection);
    cudaFree(nmb_infected_cells);
    cudaFree(sick_cell_type_count);
    for (i=NMB_CELL_TYPES-1; i>=0;  i--) {
        free(healthy_cell_type_tissue[i]);
    }
    free(healthy_cell_type_tissue);
    free(healthy_cell_type_count);


    return(0);
}







void initialize(float *ctc, char **ct)
{
    int i;
    const float cell_type_count[] = {   500000.00e05,
                                          1490.00e05,
                                          1230.00e05,
                                           806.00e05,
                                           703.00e05,
                                          1610.00e05,
                                             4.94e05,
                                         15800.00e05,
                                            84.80e05,
                                     263000000.00e05,
                                        517000.00e05,
                                      14500000.00e05,
                                         11000.00e05,
                                          7110.00e05,
                                       7530000.00e05,
                                         40000.00e05,
                                         20000.00e05,
                                        103000.00e05,
                                       2410000.00e05,
                                        963000.00e05,
                                        241000.00e05,
                                        386000.00e05,
                                        699000.00e05,
                                        290000.00e05,
                                         43200.00e05,
                                         76800.00e05,
                                       1410000.00e05,
                                         17400.00e05,
                                         33000.00e05,
                                       1370000.00e05,
                                          4490.00e05,
                                         10300.00e05,
                                      30000000.00e05,
                                       1000000.00e05,
                                         29500.00e05,
                                          2500.00e05,
                                        150000.00e05,
                                      18500000.00e05,
                                           481.00e05,
                                        329000.00e05,
                                       1370000.00e05,
                                         25800.00e05,
                                         38000.00e05,
                                         36200.00e05,
                                        167000.00e05,
                                           104.00e05,
                                         10900.00e05,
                                         11800.00e05,
                                         67600.00e05,
                                         17700.00e05,
                                         70200.00e05,
                                             8.70e05,
                                        100000.00e05,
                                      25400000.00e05};
    const char* cell_type[] = {"adipose tissue: adipocytes", 
                               "articular cartilage: femoral cartilage cells",
                               "articular cartilage: humeral head cartilage cells",
                               "articular cartilage: talus cartilage cells",
                               "biliary system: biliary ducts epithelial cells",
                               "biliary system: gallbladder epithelial cells",
                               "biliary system: gallbladder interstitial Cajal-like cells",
                               "biliary system: gallbladder smooth myocytes",
                               "biliary system: gallbladder other stromal cells",
                               "blood: erythrocytes",
                               "blood: leucocytes",
                               "blood: platelets",
                               "bone: cortical osteocytes",
                               "bone: trabecular osteocytes",
                               "bone marrow: nucleated cells",
                               "heart: connective tissue cells",
                               "heart: muscle cells",
                               "kidney: glomerulus cells",
                               "liver: hepatocytes",
                               "liver: kupffer cells",
                               "liver: stellate cells",
                               "lung bronchi bronchioles: alveolar cells type I",
                               "lung bronchi bronchioles: alveolar cells type II",
                               "lung bronchi bronchioles: alveolar macrophages",
                               "lung bronchi bronchioles: basal cells",
                               "lung bronchi bronchioles: ciliated cells",
                               "lung bronchi bronchioles: endothelial cells",
                               "lung bronchi bronchioles: goblet cells",
                               "lung bronchi bronchioles: indeterminate bronchial bronchiolar cells",
                               "lung bronchi bronchioles: interstitial cells",
                               "lung bronchi bronchioles: other bronchial bronchiolar secretory cells",
                               "lung bronchi bronchioles: preciliated cells",
                               "nervous system: glial cells",
                               "nervous system: neurons",
                               "pancreas: islet cells",
                               "skeletal muscle: muscle fibers",
                               "skeletal muscle: satellite cells",
                               "skin: dermal fibroblasts",
                               "skin: dermal mast cells",
                               "skin: epidermal corneocytes",
                               "skin: epidermal nucleate cells",
                               "skin: epidermal Langerhans cells",
                               "skin: epidermal melanocytes",
                               "skin: epidermal Merkel cells",
                               "small intestine: enterocytes",
                               "stomach: G-cells",
                               "stomach: parietal cells",
                               "suprarenal gland: medullary cells",
                               "suprarenal gland: zona fasciculata cells",
                               "suprarenal gland: zona glomerularis cells",
                               "suprarenal gland: zona reticularis cells",
                               "thyroid: clear cells",
                               "thyroid: follicular cells",
                               "vessels: endothelial cells"};

    for (i=0; i<NMB_CELL_TYPES; i++) {
        ctc[i] = cell_type_count[i];
        strcpy(ct[i], cell_type[i]);
    }           


    return;
}







void random_infection(float tot_nmb_cells, 
                      float *cell_type_count, float *infection)
{
    int i;
    float random_number, lower_bound, upper_bound;                    
    
    random_number = (float) rand() / (float) RAND_MAX;

   /*
    * so now that we got a random_number somewhere in between 0 and 1
    * we can use it to identify a particular tissue; in order to do so 
    * we consider the entire number of cells as range covering the interval 
    * from 0.0 to 1.0 and define subranges therein to represent individual
    * tissues, e.g.  
    *    0.000000 -> 0.001351...tissue 0  (i.e. 5.00e10/3.72e13 = 0.001351)
    *    0.001351 -> 0.001355...tissue 1  (i.e. 1.49e08/3.72e13 = 0.000004)
    *    0.001355 -> 0.001358...tissue 2  (i.e. 1.23e08/3.72e13 = 0.000003)
    *    .....
    * and thus we just need to walk along the NMB_CELL_TYPES intervals and
    * see whether our random_number falls into that subrange;
    */
    lower_bound = 0.0;
    upper_bound = 0.0;
    for (i=0; i<NMB_CELL_TYPES; i++) {
        infection[i] = (float) 0;
        upper_bound += cell_type_count[i] / tot_nmb_cells;
        if ((random_number >= lower_bound) && (random_number <= upper_bound)) {
           infection[i] = (float) -1;
        }
        lower_bound = upper_bound;
    }


    return;
}  







void monitor_infection_progress(float *healthy_cell_type_count, 
                                float *sick_cell_type_count,  
                                float *nmb_infected_cells)          
{
    int i, time;
    float healthy_fraction;
    dim3 thrds_per_block, blcks_per_grid;

    

   /*
    * initialize with appropriate data relevant for time 0, ie the time of
    * de-novo infection with H7N9
    */
    for (i=0; i<NMB_CELL_TYPES; i++) {
        sick_cell_type_count[i] = healthy_cell_type_count[i];
        nmb_infected_cells[i] = 0.0;
    }
    sick_cell_type_count[H7N9_TISSUE_TARGET] -= 1.0;
    nmb_infected_cells[H7N9_TISSUE_TARGET] = 1.0;
	    
   /*
    * simulate a time span of interest to follow/survey the evolution
    * of cell counts in case of H7N9 infection
    */
    for (i=1; i<NMB_TIME_STEPS_2STUDY; i++) {
        time = i * TIME_STEP_IN_HOURS;
        
       /*
        * consider the most general case of updating all tissue types 
        * in terms of current cell counts, regardless of whether or not 
        * they are directly implicated in H7N9 infection; this update
        * of all the NMB_CELL_TYPES types of different tissues may be
        * done concurrently, so a perfect task to be outsourced to the 
        * GPU, thus we need to set up an appropriate kernel execution 
        * configuration, hence one block of NMB_CELL_TYPES threads;
        */
        thrds_per_block.x = (int) NMB_CELL_TYPES;
        blcks_per_grid.x = (int) 1;
        update_nmb_infected_cells<<<blcks_per_grid, thrds_per_block>>>(time,
                                                                       sick_cell_type_count,
                                                                       nmb_infected_cells);
        cudaDeviceSynchronize();
        printf("%6d%12.3e\n", time, nmb_infected_cells[H7N9_TISSUE_TARGET]);

       /*
        * and in case we drop below a critical level of required
        * healthy cells abort and declare the organism dead
        */
	healthy_fraction = sick_cell_type_count[H7N9_TISSUE_TARGET] /  
			          healthy_cell_type_count[H7N9_TISSUE_TARGET];
        if (100.0 * healthy_fraction < (float) H7N9_TISSUE_SURVIVAL_PERCENTAGE) {
           printf("*** this organism died %6d hours post infection ***\n", time);
           printf("*** nmb_infected_cells[X] %14.6e ***\n", nmb_infected_cells[H7N9_TISSUE_TARGET]);
           printf("*** sick_cell_type_count[X] %14.6e ***\n", sick_cell_type_count[H7N9_TISSUE_TARGET]);
           printf("*** healthy_fraction %14.6e ***\n", healthy_fraction);
           break;
        }
    }


    return;
}  







