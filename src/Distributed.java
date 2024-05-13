import java.awt.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import mpi.MPI;
import mpi.MPIException;
import mpi.Status;

public class Distributed {
    static final int Width = 800;
    static final int Height = 600;
    static final int Particles = 3000;
    static final int NumCycles = 500;
    static final float particleRadius = 10; // (5pixels,5pixels)
    private static Particle[] particles;

    public static void main(String[] args) throws MPIException {
        MPI.Init(args); // initialize MPI environment using command line arguments
        int rank = MPI.COMM_WORLD.Rank(); // get rank of current process in MPI communicator
        int size = MPI.COMM_WORLD.Size(); // get total number of processes in MPI communicator

        int numParticles = Particles;
        particles = createParticles(numParticles, Width, Height);
        System.out.println("Processor " + rank + " started."); // (for testing)

        long startTime = System.currentTimeMillis(); // record start time
        for (int cycle = 0; cycle < NumCycles; cycle++) { // do simulation for each grid
            simulateParticles(particles, rank, cycle);
            exchangeBoundaryParticles(particles, rank, size);
        }
        long endTime = System.currentTimeMillis(); // record stop time
        long elapsedTime = endTime - startTime;

        // gather particle counts from all processes to rank 0
        int[] myParticleCount = new int[]{particles.length};
        int[] allParticleCounts = new int[size];
        MPI.COMM_WORLD.Gather(myParticleCount, 0, 1, MPI.INT, allParticleCounts, 0, 1, MPI.INT, 0);

        if (rank == 0) {
            int totalParticles = 0;
            for (int count : allParticleCounts) {
                totalParticles += count;
            }
            System.out.println("Total Particles: " + totalParticles);
            System.out.println("Total Cycles: " + NumCycles);
            System.out.println("Grid Width: " + Width + ", Grid Height: " + Height);
            System.out.println("Total Time: " + elapsedTime + "ms");
        }
        MPI.Finalize();
    }

    static Particle[] createParticles(int numParticles, int width, int height) {
        Map<Integer, ParticleProperties> initialParticleProperties = new HashMap<>();
        Random rand = new Random(42); // seed for random instance so positions are always the same (for testing)

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

    static void exchangeBoundaryParticles(Particle[] particles, int rank, int size) throws MPIException {
        int leftRank = (rank - 1 + size) % size;
        int rightRank = (rank + 1) % size;

        // left Exchanges
        if (leftRank >= 0) {
            int sendPosX = (rank * 100) + 1;
            int recvPosX = (leftRank * 100) + 99;
            exchangeParticles(particles, leftRank, sendPosX, recvPosX);
        }
        // right Exchanges
        if (rightRank < size) {
            int sendPosX = (rank * 100) + 99;
            int recvPosX = (rightRank * 100) + 1;
            exchangeParticles(particles, rightRank, sendPosX, recvPosX);
        }
    }

    static void exchangeParticles(Particle[] particles, int destRank, int sendPosX, int recvPosX) throws MPIException {
        if (particles[0] != null && particles[0].pos.x == sendPosX) {
            Particle boundaryParticle = particles[0]; // get boundary particle
            particles[0] = null; // remove particle from current process
            MPI.COMM_WORLD.Isend(boundaryParticle, 0, 1, MPI.OBJECT, destRank, 0); // send particle

            // receive boundary particle from neighbor
            Status status = MPI.COMM_WORLD.Probe(destRank, 0);
            int count = status.Get_count(MPI.OBJECT);
            Particle[] recvParticles = new Particle[count];
            MPI.COMM_WORLD.Recv(recvParticles, 0, count, MPI.OBJECT, destRank, 0);

            // create new particle with received information
            Particle newParticle = new Particle(recvParticles[0].pos, particleRadius, recvParticles[0].color);
            newParticle.pos.x = recvPosX; // adjust x-coordinate of new particle
            particles[0] = newParticle;
        }
    }

    static void simulateParticles(Particle[] particles, int rank, int cycle) {
        int particlesMoved = 0;
        // simulate particles within assigned grid
        for (int i = 0; i < particles.length; i++) {
            PVector previousPos = particles[i].pos.copy(); // store previous position
            particles[i].update(particles, i, Width, Height);
            if (!particles[i].pos.equals(previousPos)) {
                particlesMoved++;
            }
        }
        if (rank == 0 && cycle == 0 && particles.length > 0) {
            Particle firstParticle = particles[0];
            // (for testing)
            System.out.println("Initial location of first particle and velocity: " +
                    "X: " + firstParticle.pos.x + ", Y: " + firstParticle.pos.y +
                    ", Velocity X: " + firstParticle.vel.x + ", Velocity Y: " + firstParticle.vel.y);
        }
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