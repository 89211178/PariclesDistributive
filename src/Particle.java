import java.awt.*;
import java.util.Random;

import java.io.Serializable;
// its objects can be converted into a stream of bytes and then stored or transmitted
// to be later reconstructed to create new object with the same state

class Particle implements Serializable {
    PVector pos, vel;
    float r, spd = 0.1f, max = 2;
    Color color;

    Particle(PVector pos, float r, Color color) {
        this.pos = pos;
        this.r = r;
        this.color = color;
        vel = new PVector(new Random().nextFloat(), new Random().nextFloat()); // initialize random velocity
    }

    void update(Particle[] p, int i, int width, int height) {
        pos.add(vel);
        repelFromWalls(width, height);
        randomizeVelocity();
        calculateForce(p, i);
    }

    void repelFromWalls(int width, int height) {
        if (pos.x < 0 || pos.x > width) vel.x *= -1; // left or right wall, reverse x-velocity
        if (pos.y < 0 || pos.y > height) vel.y *= -1; // top or bottom wall, reverse y-velocity
    }

    void randomizeVelocity() {
        Random rand = new Random();
        // randomize x and y velocity within limits
        vel.x = constrain(vel.x + rand.nextFloat() * 2 * spd - spd, -max, max);
        vel.y = constrain(vel.y + rand.nextFloat() * 2 * spd - spd, -max, max);
    }

    void calculateForce(Particle[] p, int i) {
        float attractionForce = 0.01f, repulsionForce = -0.01f;

        for (int j = i + 1; j < p.length; j++) {
            float dist = pos.dist(p[j].pos); // calculate distance
            if (dist < r) { // distance is less than sum of radii
                float force = p[j].color.equals(color) ? attractionForce : repulsionForce; // determine force based on color
                float ang = pos.atan2(p[j].pos.y - pos.y, p[j].pos.x - pos.x); // calculate angle
                vel.x += force * Math.cos(ang);
                vel.y += force * Math.sin(ang);
            }
        }
    }

    float constrain(float value, float min, float max) {
        return Math.min(Math.max(value, min), max);
    }
}