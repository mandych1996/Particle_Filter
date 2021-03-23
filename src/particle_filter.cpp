/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

#define N_particles 200;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = N_particles; 
  vector<Particles> particles(N_particles);

  default_random_engine gen;
  normal_distribution<double> x_dist(x, std[0]);
  normal_distribution<double> y_dist(y, std[1]);
  normal_distribution<double> theta_dist(theta, std[2]);

  for(int i =0; i <weights.size(); i++){
    particles[i].id = i;
    particles[i].x = x_dist(gen);
    particles[i].y = y_dist(gen);
    particles[i].theta = theta_dist(gen);
    particles[i].weight = 1.0;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  default_random_engine gen;
  normal_distribution<double> x_dist(x, std[0]);
  normal_distribution<double> y_dist(y, std[1]);
  normal_distribution<double> theta_dist(theta, std[2]);

  for(auto &p : particles){
    if(fabs(yaw_rate)<0.00001){
      p.x += velocity * delta_t *cos(p.theta);
      p.y += velocity * delta_t *sin(p.theta);
    } else {
      p.x += (velocity/yaw_rate)* (sin(p.theta +yaw_rate*delta_t)-sin(p.theta));
      p.y += (velocity/yaw_rate)* (cos(p.theta)-sin(p.theta +yaw_rate*delta_t));
    }
    p.x += x_dist(gen);
    p.y += y_dist(gen);
    p.theta += theta_dist(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  for(auto const &o : observations){
    double min_dist = numeric_limits<double>::max();
    int landmark_id = -1;
    for(auto &pred_landmarks : predicted){
      double d = dist(o.x, o.y, pred_landmarks.x, pred_landmarks.y);
      if(d < min_dist){
        min_dist = d;
        landmark_id = p.id;
      }
    }
    assert(landmark_id != -1);
    o.id = landmark_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  double sigma_xx = pow(std_landmark[0], 2.0);
  double sigma_yy = pow(std_landmark[1], 2.0);
  double std_xy = std_landmark[0]*std_landmark[1];
  double sensor_r2 = pow(senosr_range, 2.0);

  for(auto &p : particles){
    // Transform the observations into MAP'S coordinate system
    vecotor<LandmarksObs> observations_map(observations.size());
    int i =0;
    for(auto const &o : observations){
      observations_map[i] = {
        .id = -1;
        .x = p.x + o.x *cos(p.theta) - o.y *sin(p.theta);
        .y = p.y + o.x *sin(p.theta) + o.y *cos(p.theta);
      }
      i++;
    }

    // Find landmarks within sensor range
    vector<LandmarksObs> landmarks_in_range;
    for(auto const &l : map_landmarks.landmark_list){
      double x_distance_square = pow(p.x-l.x, 2.0);
      double y_distance_square = pow(p.y-l.y, 2.0);
      if(x_distance_square + y_distance_square < sensor_r2){
        landmarks.push_back(LandmarkObs{
          l.id_i,
          static_cast<double>(l.x_f),
          static_cast<double>(l.y_f)
          });
      }
    }
    assert(!landmarks_in_range.empty());
    dataAssociation(landmarks_in_range, observations_map);

    // Update particle's weight
    double probability_density;
    p.weight = 1.0;
    for(auto &o_map : observations_map){
      int nearest_landmark_id = map_landmarks.landmark_list[o.id - 1].id_i; // Landmark_id starting from 1
      double nearest_landmark_x = map_landmarks.landmark_list[o.id - 1].x_f;
      double nearest_landmark_y = map_landmarks.landmark_list[o.id - 1].y_f;
      associations.push_back(nearest_landmark_id);
      sense_x.push_back(nearest_landmark_x);
      sense_y.push_bakc(nearest_landmark_y);

      double delta_x2 = pow(o.x -nearest_landmark_x, 2.0);
      double delta_y2 = pow(o.y -nearest_landmark_y, 2.0);
      probability_density = 1/(2*M_PI*std_xy)*exp(-(delta_x2/(2*sigma_xx) +delta_y2/(2*sigma_yy)));
      p.weight *= probability_density;
      
    }
    SetAssociations(p, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  default_random_engine gen;
  discrete_distribution<int> p_dist(weights.begin(), weights.end());

  vector<Particle> resampled_particles(particles.size());
  for(auto &resampled_p : resampled_particles){
    int index = p_dist(gen);
    resampled_p = particles[index];
  }
  particles = move(resampled_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}