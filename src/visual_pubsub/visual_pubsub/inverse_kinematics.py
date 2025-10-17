import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point 
import numpy as np

class InverseKinematics(Node):

    def __init__(self):
        super().__init__('inverse_kinematics_pure_jacobian')
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # 1. PARÁMETROS DEL ROBOT Y LÍMITES
        self.l1 = 2.0  
        self.l2 = 1.5  
        self.l3 = 0.8
        
        # Límites Articulares
        self.q_min = np.array([-np.pi, -np.pi/2, -np.pi/2])
        self.q_max = np.array([np.pi, np.pi/2, np.pi/2])

        # 2. PUNTO OBJETIVO FIJO 
        self.target_pos = np.array([1.2, 0.5, 0.0])  
        
        # 3. PARÁMETROS DE CORRECCIÓN IK (sin control P explícito)
        self.q = np.array([-0.22, 0.7, 0.03])  # Ángulos iniciales
        
        # El paso de integración fijo (Delta_t) se mantiene
        self.step_size = 0.02       
        self.tolerance = 0.015       
        
        # Parámetros del Amortiguamiento Dinámico (lambda)
        self.k_min = 0.01   
        self.k_max = 8.0    
        
        # La ganancia de escalado ahora está implícita en la norma del error o en el step_size
        self.global_gain_factor = 1.0 # Factor de ganancia global para ajustar la velocidad

        self.timer = self.create_timer(0.02, self.update_joints) # 50 Hz (20ms)

        self.get_logger().info(f"Nodo IK jacobiano puro iniciado. Objetivo: {self.target_pos}")

    
    # --- FUNCIONES DE CINEMÁTICA ---
    def forward_kinematics(self, q):
        q1, q2, q3 = q
        x = self.l3 * np.sin(q1) * np.sin(q3) + self.l3 * np.cos(q1) * np.cos(
            q2) * np.cos(q3) + self.l2 * np.cos(q1) * np.cos(q2)
        y = self.l3 * np.sin(q1) * np.cos(q2) * np.cos(q3) + self.l2 * np.sin(
            q1) * np.cos(q2) - self.l3 * np.sin(q3) * np.cos(q1)
        z = self.l3 * np.sin(q2) * np.cos(q3) + self.l2 * np.sin(q2) + self.l1
        return np.array([x, y, z])

    def jacobian(self, q):
        q1, q2, q3 = q
        j11 = -self.l3 * np.sin(q1) * np.cos(q2) * np.cos(q3) - self.l2 * np.sin(q1) * np.cos(q2) + self.l3 * np.sin(q3) * np.cos(q1)
        j12 = -self.l3 * np.sin(q2) * np.cos(q1) * np.cos(q3) - self.l2 * np.sin(q2) * np.cos(q1)
        j13 = self.l3 * np.sin(q1) * np.cos(q3) - self.l3 * np.sin(q3) * np.cos(q1) * np.cos(q2)
        j21 = self.l3 * np.sin(q1) * np.sin(q3) + self.l3 * np.cos(q1) * np.cos(q2) * np.cos(q3) + self.l2 * np.cos(q1) * np.cos(q2)
        j22 = -self.l3 * np.sin(q1) * np.sin(q2) * np.cos(q3) - self.l2 * np.sin(q1) * np.sin(q2)
        j23 = -self.l3 * np.sin(q1) * np.sin(q3) * np.cos(q2) - self.l3 * np.cos(q1) * np.cos(q3)
        j31 = 0
        j32 = self.l3 * np.cos(q2) * np.cos(q3) + self.l2 * np.cos(q2)
        j33 = -self.l3 * np.sin(q2) * np.sin(q3)
        return np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]])

    def apply_joint_limits(self, q):
        return np.clip(q, self.q_min, self.q_max)
    # -----------------------------------------------

    def update_joints(self):
        current_pos = self.forward_kinematics(self.q)
        error = self.target_pos - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm > self.tolerance:
            
            # 1. Definir un factor de escalamiento basado en la norma del error.
            # Esto funciona como una ganancia implícita que desaparece cerca del objetivo.
            # Se usa el error_norm para que el paso sea proporcional a la distancia.
            scaling_factor = self.global_gain_factor * error_norm 
            
            # Opcionalmente, limitar la velocidad máxima de movimiento (ej: a 0.5 m/s)
            max_speed = 0.5
            if scaling_factor > max_speed:
                scaling_factor = max_speed
            
            # 2. Calcular el factor de amortiguamiento dinámico (lambda)
            error_ratio = np.clip(error_norm / self.tolerance, 0.0, 1.0)
            inverse_error_ratio = 1.0 - error_ratio
            damping_factor_dyn = self.k_min + (self.k_max - self.k_min) * (inverse_error_ratio)**3
            damping_factor_dyn = np.clip(damping_factor_dyn, self.k_min, self.k_max)
            
            J = self.jacobian(self.q)

            JtJ = J.T @ J
            damping_sq = (damping_factor_dyn ** 2) * np.eye(JtJ.shape[0])
            
            # 3. Calcular el cambio articular (Delta_q)
            # dq = (J^T J + lambda^2 I)^-1 * J^T * (Error de Posición)
            dq = np.linalg.solve(JtJ + damping_sq, J.T) @ error
            
            # 4. Integración: Aplicar el paso escalado
            # q_nuevo = q_actual + (dq * factor_de_escalado * Delta_t)
            # Normalizamos dq primero para que el factor de escalado controle la magnitud
            dq_normalized = dq / np.linalg.norm(dq) if np.linalg.norm(dq) > 1e-6 else dq
            
            self.q += dq_normalized * scaling_factor * self.step_size
            self.q = self.apply_joint_limits(self.q)
            
        else:
            self.get_logger().info("Objetivo alcanzado")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['q1', 'q2', 'q3'] 
        msg.position = self.q.tolist()
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()