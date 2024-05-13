import java.io.Serializable;
// its objects can be converted into a stream of bytes and then stored or transmitted
// to be later reconstructed to create new object with the same state

class PVector implements Serializable {
    // x and y coordinate of the vector
    float x, y;

    PVector(float x, float y) {
        this.x = x;
        this.y = y;
    }

    // Euclidean distance
    float dist(PVector other) {
        float dx = this.x - other.x;
        float dy = this.y - other.y;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    void add(PVector v) {
        this.x += v.x;
        this.y += v.y;
    }

    PVector copy() {
        return new PVector(this.x, this.y);
    }

    // return the arctangent of the ratio of y to x
    float atan2(float y, float x) {
        return (float) Math.atan2(y, x);
    }
}