package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import vtk.vtkActor;
import vtk.vtkPolyDataMapper;
import vtk.vtkStreamTracer;
import vtk.vtkStructuredGrid;
import vtk.vtkPoints;
import vtk.vtkRenderer;
import vtk.vtkRenderWindow;
import vtk.vtkRenderWindowInteractor;
import vtk.vtkArrowSource;
import vtk.vtkGlyph3D;
import vtk.vtkLookupTable;
import vtk.vtkVersion;

/**
 * Plots 3D streamlines from azimuth and zenith JCuda arrays using VTK.
 * 
 * @author E. Dov Neimand
 */
public class StreamLinePlotter {

    private final FStrideArray3d azimuth;
    private final FStrideArray3d zenith;
    private final Handle handle;

    /**
     * Constructs a streamline plotter.
     *
     * @param azimuth Azimuth angles stored as a JCuda FStrideArray3d.
     * @param zenith Zenith angles stored as a JCuda FStrideArray3d.
     * @param handle GPU handle for memory management.
     */
    public StreamLinePlotter(FStrideArray3d azimuth, FStrideArray3d zenith, Handle handle) {
        this.azimuth = azimuth;
        this.zenith = zenith;
        this.handle = handle;
    }
//
//    /**
//     * Converts azimuth and zenith angles to velocity vectors and plots streamlines.
//     */
//    public void plotStreamlines() {
//        int nx = azimuth.shape()[0];
//        int ny = azimuth.shape()[1];
//        int nz = azimuth.shape()[2];
//
//        float[] azimuthData = azimuth.get(handle);
//        float[] zenithData = zenith.get(handle);
//
//        vtkStructuredGrid grid = new vtkStructuredGrid();
//        grid.SetDimensions(nx, ny, nz);
//
//        vtkPoints points = new vtkPoints();
//        vtkFloatArray vectors = new vtkFloatArray();
//        vectors.SetNumberOfComponents(3);
//        vectors.SetName("Velocity");
//
//        for (int k = 0; k < nz; k++) {
//            for (int j = 0; j < ny; j++) {
//                for (int i = 0; i < nx; i++) {
//                    int index = (i * ny * nz) + (j * nz) + k;
//                    float azimuth = azimuthData[index];
//                    float zenith = zenithData[index];
//
//                    float vx = (float) (Math.sin(zenith) * Math.cos(azimuth));
//                    float vy = (float) (Math.sin(zenith) * Math.sin(azimuth));
//                    float vz = (float) Math.cos(zenith);
//
//                    points.InsertNextPoint(i, j, k);
//                    vectors.InsertNextTuple3(vx, vy, vz);
//                }
//            }
//        }
//
//        grid.SetPoints(points);
//        grid.GetPointData().SetVectors(vectors);
//
//        vtkStreamTracer tracer = new vtkStreamTracer();
//        tracer.SetInputData(grid);
//        tracer.SetIntegrationDirectionToForward();
//        tracer.SetMaximumPropagation(200);
//        tracer.SetInitialIntegrationStep(0.1);
//        tracer.SetComputeVorticity(true);
//
//        vtkPolyDataMapper mapper = new vtkPolyDataMapper();
//        mapper.SetInputConnection(tracer.GetOutputPort());
//
//        vtkActor actor = new vtkActor();
//        actor.SetMapper(mapper);
//
//        vtkRenderer renderer = new vtkRenderer();
//        vtkRenderWindow renderWindow = new vtkRenderWindow();
//        renderWindow.AddRenderer(renderer);
//        vtkRenderWindowInteractor interactor = new vtkRenderWindowInteractor();
//        interactor.SetRenderWindow(renderWindow);
//
//        renderer.AddActor(actor);
//        renderer.SetBackground(0.1, 0.1, 0.1);
//        renderWindow.Render();
//        interactor.Start();
//    }
    
    public static void main(String[] args) {
        vtkVersion vtkVersion = new vtkVersion();
        System.out.println("VTK Version: " + vtkVersion.GetVTKVersion());
    }
}
