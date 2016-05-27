#include <iostream>
#include <vector>
#include "Python.h"
#include "numpy/arrayobject.h"

void convex_hull(double const* const points, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index);
void inside_hull(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, bool *inside);

void inside_hull_iterative(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, bool *inside);
 
void inside_hull_sorted(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, bool *inside);

static PyObject *graham_scan_inside_hull(PyObject *dummy, PyObject *args)
{
    /* Perform a Graham scan and check which points are inside the hull
     * 
     * Input: to-hull points, test points
     * Output: hull, bools if inside
     */
    
    /* Variable declaration:
     * 
     */
    // SIGNAL
    double *points;
    double *testpoints;
    bool *inside;
    int n_points;
    int t_points;
    int dims_inside[2];
    int dims_hull[2];
    PyArrayObject *hull_arr;
    PyArrayObject *inside_arr;
    
    PyObject *arg1=NULL, *arg2=NULL;
    PyObject *arr1=NULL, *arr2=NULL;
    
    /* Argument parsing:
     * We need to parse the arguments, and then put the data
     * in arrays.
     */
    // Parse arguments
    if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) return NULL;

    // Put the arguments in arrays
    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr2 == NULL) return NULL;
    
    
    points = (double *)PyArray_DATA(arr1); // pointer to data.
    testpoints = (double *)PyArray_DATA(arr2); // pointer to data.
    n_points = PyArray_DIM(arr1, 0);
    t_points = PyArray_DIM(arr2, 0);
     
    // Create convex hull
    std::vector<double> myhull;
    int h_points = 100;
    int turn_index = 0;
    
    inside = new bool[t_points];
    
    convex_hull(points, n_points, myhull, h_points, turn_index);    
    
    // The following are different versions of the algorithm
    //inside_hull(testpoints, t_points, myhull, h_points, turn_index, inside);
    //inside_hull_iterative(testpoints, t_points, myhull, h_points, turn_index, inside);
    inside_hull_sorted(testpoints, t_points, myhull, h_points, turn_index, inside);
    
    // Create the output arrays: hull and boolean inside
    dims_hull[0] = h_points;
    dims_hull[1] = 2;
    hull_arr = (PyArrayObject *)PyArray_FromDims(2, dims_hull, PyArray_DOUBLE);
    dims_inside[0] = t_points;
    inside_arr = (PyArrayObject *)PyArray_FromDims(1, dims_inside, PyArray_BOOL);
    
    for (int i=0;i<t_points;i++){
        ((bool *)inside_arr->data)[i] = inside[i];
    }
    for (int i=0;i<h_points;i++){
        ((double *)hull_arr->data)[2*i] = myhull[2*i];
        ((double *)hull_arr->data)[2*i+1] = myhull[2*i+1];
    }
    
    delete inside;
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    
    // Return tuple of arrays
    PyObject *tupleresult = PyTuple_New(2);
    PyTuple_SetItem(tupleresult, 0, PyArray_Return(hull_arr));
    PyTuple_SetItem(tupleresult, 1, PyArray_Return(inside_arr));
    return tupleresult;
    //return Py_BuildValue("OO", hull_arr, inside_arr);
    //return Py_BuildValue("dd", 3.0, 2.0);
}


int turn(double p1, double p2, double q1, double q2, double r1, double r2)
{
    /* Compute which way three points "bend".
     * 
     * 
     */
    double turn_value;
    int turn_flag = 0;
    
    turn_value = (q1-p1) * (r2-p2) - (r1-p1)*(q2-p2);
    
    if (turn_value < 0.0){
        turn_flag = -1;
    }
    else if (turn_value > 0.0){
        turn_flag =  1;
    }
    
    return turn_flag;   
}
    

void convex_hull(double const* const points, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index)
/* Perform a Graham scan of a set of points to find the convex hull.
 * 
 * In the first run, the lower part of the convex hull is found, in the second
 * part the upper part is found.
 * 
 * input: array of points [2*N]
 *        empty vector and initial guess of hull points
 * output: array of hull points [2, M] (M to be determined by this function)
 *         index of "turning point".
 */
{
    
    myhull.reserve(2*h_points);
    int msize = 0;
    int turn_flag;
    int xp = 0; // index to x-coordinate of points array
    int yp = n_points; // index to y-coordinate of points array
    int xhi1, yhi1, xhi2, yhi2;
    
    h_points = 2;
    
    // add first two points:
    myhull.push_back(points[0]);
    myhull.push_back(points[1]);
    myhull.push_back(points[2]);
    myhull.push_back(points[3]);
    
    // first run
    for(int i=2; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        
        // if there are more than 2 (x,y) points in hull, check if the last one
        // turns in the correct direction
        if (myhull.size()/2 >= 2) {
            
            // if we turn the wrong way, we need to remove the point
            // from the hull. Then we need to check the turn for the
            // following two last points of the hull, untill exhaustion
            while (myhull.size()>=4){
            //for(int j=0; j<myhull.size()/2+3; j++){
                xhi1 = myhull.size()-4;
                yhi1 = myhull.size()-3;
                xhi2 = myhull.size()-2;
                yhi2 = myhull.size()-1;
                
                turn_flag = turn(myhull[xhi1], myhull[yhi1],
                                myhull[xhi2], myhull[yhi2],
                                points[xp], points[yp]);
                
                // Woopsy; if turn_flag does not equal one, it is not part of the
                // convex hull
                if (turn_flag < 1){
                    myhull.pop_back();
                    myhull.pop_back();
                    if (myhull.size()/2<2){
                        break;
                    }
                }
                else{
                    break;
                }
            }
        }
        
        // Add this point to the hull if it satisfies the convex hull criterion
        if ((myhull.size()/2==2) || (myhull[myhull.size()-2]!=points[xp]) || (myhull[myhull.size()-1]!=points[yp])){
            myhull.push_back(points[xp]);
            myhull.push_back(points[yp]);
        }
    }
    
    // Remember the turning point, and the current size of the hull (we need
    // to start over the search, but append the results to the previous vector)
    turn_index = myhull.size()/2;
    h_points = myhull.size()-2;
    
    // Second run
    for(int i=n_points-2; i>=0; i--){
        xp = 2*i;
        yp = 2*i+1;
        
        // if there are more than 2 (x,y) points in hull, check if the last one
        // turns in the correct direction
        msize = myhull.size()-h_points;
        if (msize/2 >= 2) {
            
            // if we turn the wrong way, we need to remove the point
            // from the hull. Then we need to check the turn for the
            // following two last points of the hull, untill exhaustion
            while (myhull.size()>=h_points){
                xhi1 = myhull.size()-4;
                yhi1 = myhull.size()-3;
                xhi2 = myhull.size()-2;
                yhi2 = myhull.size()-1;
                
                turn_flag = turn(myhull[xhi1], myhull[yhi1],
                                myhull[xhi2], myhull[yhi2],
                                points[xp], points[yp]);
                // Woopsy; if turn_flag does not equal one, it is not part of the
                // convex hull
                if (turn_flag < 1){
                    myhull.pop_back();
                    myhull.pop_back();
                    
                    if ((myhull.size()-h_points)/2<2){
                        break;
                    }
                }
                else{
                    break;
                }
            }
        }
        
        msize = myhull.size() - h_points;
        // Add this point to the hull if it satisfies the convex hull criterion
        if ((msize/2==2) || (myhull[msize-2]!=points[xp]) || (myhull[msize-1]!=points[yp])){
            myhull.push_back(points[xp]);
            myhull.push_back(points[yp]);
        }
    }
    
    h_points = myhull.size()/2;
        
}


void inside_hull(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, bool *inside)
/* Check which testpoints are inside a convex hull.
 * 
 */
{
    int xp = 0;
    int yp = 0;
    int myturn = 0;
    bool skip = false;
    int current_index = 0;
    int current_index2 = 0;
    double miny_hull =1e300;
    double maxy_hull = -1e300;
    
    // look for minimum and maximum y value, for chop-offs
    for (int i=0; i<h_points;i++){
        current_index = 2*i+1;
        if (myhull[current_index] < miny_hull){
            miny_hull = myhull[current_index];
        }
        if (myhull[current_index] > maxy_hull){
            maxy_hull = myhull[current_index];
        }
    }
    
    
    for (int i=0; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        inside[i] = true;
        
        // entirely on the left or right: the test point is outside the hull
        if ((testpoints[xp]<myhull[0]) || (myhull[2*turn_index-2]<testpoints[xp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = false;
            continue;
        }
        if ((testpoints[yp]<miny_hull) || (maxy_hull<testpoints[yp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = false;
            continue;
        }
        
        
        // below the lower hull? then outside the hull!
        //std::cout << "# test turn point: " << testpoints[xp]<<" "<< testpoints[yp] << std::endl;
        for (int j=2; j<h_points; j++){
            current_index = 2*(j-2);
            current_index2 = 2*(j-1);
            myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[current_index2], myhull[current_index2+1],
                    testpoints[xp], testpoints[yp]);
            //std::cout << "# .. " << myhull[2*(j-2)] << " "<< myhull[2*(j-2)+1]<< " "<<myhull[2*(j-1)]<< " "<<myhull[2*(j-1)+1] << " // "<<myturn << std::endl;
            if (myturn<1){
                inside[i] = false;
                skip = true;
                break;
            }
        }
        if (skip){
            skip = false;
            continue;
        }
        
        // last points closes in on itself
        current_index = 2*(h_points-2);
        myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[0], myhull[1],
                    testpoints[xp], testpoints[yp]);
        if (myturn<1){
            inside[i] = false;
        }
    }
    
}


int binary_search(std::vector<double>& A, int npoints, double key, int imin)
{
    int imax = npoints-1;
    int imid = 0;
    
    if (key < A[2*imin]){
        return imin;
    }
    else if (key > A[2*imax]){
        return imax;
    }
    
    while (imax >= imin){
        // calculate the midpoint for roughly equal partition
        imid = imin + ((imax-imin)/2);
        // key found at index imid
        if (A[2*imid] == key){
            return imid;
        }
        // determine which subarray to search
        else if (A[2*imid] < key){
            // change min index to search upper subarray
            imin = imid + 1;
        }
        // change max index to search lower subarray
        else{
            imax = imid - 1;
        }
    }
    // key was not found
    return imin;
}


void inside_hull_sorted(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, bool *inside)
/* Check which testpoints are inside a convex hull.
 * 
 */
{
    int xp = 0;
    int yp = 0;
    int imid1 = 0;
    int myturn = 0;
    int current_index = 0;
    int current_index2 = 0;
    double miny_hull =1e300;
    double maxy_hull = -1e300;
    std::vector<double> imyhull; // uppper (inverted hull)
    
    // create a smaller version of the hull that only contains the upper
    // part, but in reversed order (which means from smaller x to larger x)
    
    imyhull.reserve(2*(h_points-turn_index+1));
    for (int i=h_points-1; i>=turn_index-1;i--){
        imyhull.push_back(myhull[2*i]);
        imyhull.push_back(myhull[2*i+1]);
    }
    
    
    // look for minimum and maximum y value, for easy chop-offs
    for (int i=0; i<h_points;i++){
        current_index = 2*i+1;
        if (myhull[current_index] < miny_hull){
            miny_hull = myhull[current_index];
        }
        if (myhull[current_index] > maxy_hull){
            maxy_hull = myhull[current_index];
        }
    }
    
    
    for (int i=0; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        inside[i] = true;
        
        // entirely on the left or right: the test point is outside the hull
        if ((testpoints[xp]<=myhull[0]) || (myhull[2*turn_index-2]<=testpoints[xp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = false;
            continue;
        }
        if ((testpoints[yp]<=miny_hull) || (maxy_hull<=testpoints[yp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = false;
            continue;
        }
        
        // below the lower hull? then outside the hull!
        // the n_points are sorted, so we can remember the best position
        // for the next run
        imid1 = binary_search(myhull, turn_index-1, testpoints[xp], 0);
        current_index = 2*(imid1-1);
        current_index2 = 2*(imid1);
        myturn = turn(myhull[current_index], myhull[current_index+1],
                      myhull[current_index2], myhull[current_index2+1],
                      testpoints[xp], testpoints[yp]);
        if (myturn<1){
            inside[i] = false;
            continue;
        }
        
        // above the upper hull? then outside the hull!
        imid1 = binary_search(imyhull, imyhull.size()/2, testpoints[xp], 0);
        current_index = 2*(imid1-1);
        current_index2 = 2*(imid1);
        myturn = turn(imyhull[current_index], imyhull[current_index+1],
                      imyhull[current_index2], imyhull[current_index2+1],
                      testpoints[xp], testpoints[yp]);
        if (myturn>-1){
            inside[i] = false;
        }
    }
    
}

void inside_hull_iterative(double const* const testpoints, int n_points,
                 std::vector<double>& myhull, int& h_points,
                 int& turn_index, bool *inside)
/* Check which testpoints are inside a convex hull.
 * 
 */
{
    int xp = 0;
    int yp = 0;
    int myturn = 0;
    bool skip = false;
    bool *inside_treated;
    int resolution = 0;
    int current_index = 0;
    int current_index2 = 0;
    int split = 1;
    double miny_hull =1e300;
    double maxy_hull = -1e300;
    int total = 0;
    
    inside_treated = new bool[n_points];
    
    // look for minimum and maximum y value, for chop-offs
    for (int i=0; i<h_points;i++){
        current_index = 2*i+1;
        if (myhull[current_index] < miny_hull){
            miny_hull = myhull[current_index];
        }
        if (myhull[current_index] > maxy_hull){
            maxy_hull = myhull[current_index];
        }
    }
    
    for (int i=0; i<n_points; i++){
        inside[i] = true;
        inside_treated[i] = false;
    }
    
    
    resolution = h_points/(4*split);
        
        
    for (int i=0; i<n_points; i++){
        if (inside_treated[i]){
            continue;
        }
        xp = 2*i;
        yp = 2*i+1;
        
        // below the lower hull? then outside the hull!
        for(int j=2;j<(4*split)+1;j++){
            current_index = 2*(j-2)*resolution;
            current_index2 = 2*(j-1)*resolution;
                            
            myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[current_index2], myhull[current_index2+1],
                    testpoints[xp], testpoints[yp]);
            if (myturn<1){
                skip = true;
                break;
            }                        
        }

        if (skip){
            skip = false;
            continue;
        }
        // last points closes in on itself
        current_index = 2*(4*split-1)*resolution;
        myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[0], myhull[1],
                    testpoints[xp], testpoints[yp]);
        
        if (myturn<1){
            continue;
        }
        // If we arrived here, the point is inside the hull!
        inside_treated[i] = true;
        total = total+1;
        //std::cout << "split c" << split << std::endl;
    }
    
    
    
    
    
    for (int i=0; i<n_points; i++){
        xp = 2*i;
        yp = 2*i+1;
        
        if (inside_treated[i]){
            continue;
        }
        
        // entirely on the left or right: the test point is outside the hull
        if ((testpoints[xp]<myhull[0]) || (myhull[2*turn_index-2]<testpoints[xp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = false;
            continue;
        }
        if ((testpoints[yp]<miny_hull) || (maxy_hull<testpoints[yp])){
            //std::cout << "#"<< myhull[0] << "<" << testpoints[xp] << "<" << myhull[2*turn_index-2] << std::endl;
            inside[i] = false;
            continue;
        }
        
        
        // below the lower hull? then outside the hull!
        //std::cout << "# test turn point: " << testpoints[xp]<<" "<< testpoints[yp] << std::endl;
        for (int j=2; j<h_points; j++){
            current_index = 2*(j-2);
            current_index2 = 2*(j-1);
            myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[current_index2], myhull[current_index2+1],
                    testpoints[xp], testpoints[yp]);
            //std::cout << "# .. " << myhull[2*(j-2)] << " "<< myhull[2*(j-2)+1]<< " "<<myhull[2*(j-1)]<< " "<<myhull[2*(j-1)+1] << " // "<<myturn << std::endl;
            if (myturn<1){
                inside[i] = false;
                skip = true;
                break;
            }
        }
        if (skip){
            skip = false;
            continue;
        }
        
        // last points closes in on itself
        current_index = 2*(h_points-2);
        myturn = turn(myhull[current_index], myhull[current_index+1],
                    myhull[0], myhull[1],
                    testpoints[xp], testpoints[yp]);
        if (myturn<1){
            inside[i] = false;
        }
    }
    delete inside_treated;
    
}
                 

/*
int main()
{   
     
     double array[2000];
     double testarray[2000];
     double x, y, u, v;
     bool inside[1000];
     std::string line;
     std::ifstream myfile;
     myfile.open("test_random.txt");
     
     for(int i=0;i<1000;i++){
         
         myfile >> x >> y >> u >> v;
         array[i] = x;
         array[i+1000] = y;
         testarray[i] = u;
         testarray[i+1000] = v;
         
     }
     
     std::vector<double> myhull;
     int h_points = 10;
     int turn_index = 0;
     convex_hull(array, 1000, myhull, h_points, turn_index);
     inside_hull(testarray, 1000, myhull, h_points, turn_index, inside);
     
     
     // Attach the second run results to the first run results
     // bla bla
     for (int i=0; i<h_points; i++){
        std::cout << myhull[2*i] << " " << myhull[2*i+1] << std::endl;
     }
     std::cout << h_points << " " << turn_index << std::endl;
     
     std::cout << "===" << std::endl;
     
     for (int i=0; i<1000; i++){
       std::cout << inside[i] << std::endl;
     }
     return 0;
}
*/





// register all functions
static PyMethodDef ceclipseMethods[] = {
  {"graham_scan_inside_hull",  graham_scan_inside_hull}, //python name, C name
  {NULL, NULL, 0, NULL} //required ending
};



#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "ceclipse",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        ceclipseMethods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif



PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_ceclipse(void)
#else
initceclipse(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  (void) PyModule_Create(&moduledef);
#else
  (void) Py_InitModule3("ceclipse", ceclipseMethods,"ceclipse doc");
  import_array();
#endif
}

