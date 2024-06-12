import java.awt.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import mpi.MPI;
import mpi.MPIException;

public class Distributed {
    static final int Width = 800;
    static final int Height = 600;
    static final int Particles = 3000;
    static final int NumCycles = 500;
    static final float particleRadius = 5; // (2.5pixels,2.5pixels)
    private static Particle[] particles;

    public static void main(String[] args) throws MPIException {
        MPI.Init(args);
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        System.out.println("Processor " + rank + " started");

        if (rank == 0) {
            particles = createParticles(Particles, Width, Height);
        } else {
            particles = new Particle[Particles];
        }

        // broadcast initial state of particles to all processes
        MPI.COMM_WORLD.Bcast(particles, 0, Particles, MPI.OBJECT, 0);

        long startTime = System.currentTimeMillis();
        for (int cycle = 0; cycle < NumCycles; cycle++) {
            simulateParticles(particles, rank, size);
            gather_broadcastParticles(size); // gather updated particles, perform broadcasting, and synchronize particles across processes
        }
        long endTime = System.currentTimeMillis();
        long elapsedTime = endTime - startTime;

        if (rank == 0) {
            System.out.println("Total Particles: " + Particles);
            System.out.println("Total Cycles: " + NumCycles);
            System.out.println("Grid Width: " + Width + ", Grid Height: " + Height);
            System.out.println("Total Time: " + elapsedTime + "ms");
        }
        MPI.Finalize();
    }

    static Particle[] createParticles(int numParticles, int width, int height) {
        Map<Integer, ParticleProperties> initialParticleProperties = new HashMap<>();
        Random rand = new Random(42);

        for (int i = 0; i < numParticles; i++) {
            PVector pos = new PVector(rand.nextInt(width), rand.nextInt(height));
            PVector speed = new PVector(rand.nextFloat() * 10 - 5, rand.nextFloat() * 10 - 5);
            Color color = rand.nextBoolean() ? Color.RED : Color.BLUE;
            initialParticleProperties.put(i, new ParticleProperties(pos, speed, color));
        }
        return inputParticles(initialParticleProperties);
    }

    static Particle[] inputParticles(Map<Integer, ParticleProperties> properties) {
        Particle[] particles = new Particle[properties.size()];
        int index = 0;
        for (ParticleProperties property : properties.values()) {
            particles[index] = new Particle(property.pos, particleRadius, property.color);
            index++;
        }
        return particles;
    }

    static void simulateParticles(Particle[] particles, int rank, int size) {
        int particlesPerProcess = Particles / size;
        int startIndex = particlesPerProcess * rank;
        int endIndex = startIndex + particlesPerProcess;
        //System.out.println("Rank " + rank + " is responsible for particles from " + startIndex + " to " + (endIndex - 1));
        for (int i = startIndex; i < endIndex; i++) {
            particles[i].update(particles, i, Width, Height);
                //System.out.println("Rank: " + rank + " Pos: X: " + particles[i].pos.x + " Y: " + particles[i].pos.y +
                //                   " Vel: X: " + particles[i].vel.x + " Y: " + particles[i].vel.y); // (for testing)
            }
        }

    static void gather_broadcastParticles(int size) throws MPIException {
        int particlesPerProcess = Particles / size;
        Particle[] subParticles = new Particle[particlesPerProcess];

        // each process fills subParticles with its portion of particles
        int rank = MPI.COMM_WORLD.Rank();
        int startIndex = rank * particlesPerProcess;
        System.arraycopy(particles, startIndex, subParticles, 0, particlesPerProcess);

        // gather updated particles
        MPI.COMM_WORLD.Gather(subParticles, 0, particlesPerProcess, MPI.OBJECT, particles, 0, particlesPerProcess, MPI.OBJECT, 0);

        // broadcast updated particles
        MPI.COMM_WORLD.Bcast(particles, 0, Particles, MPI.OBJECT, 0);
    }

    // helper class to store initial particle properties
    private static class ParticleProperties {
        PVector pos;
        PVector speed;
        Color color;

        ParticleProperties(PVector pos, PVector speed, Color color) {
            this.pos = pos;
            this.speed = speed;
            this.color = color;
        }
    }
}