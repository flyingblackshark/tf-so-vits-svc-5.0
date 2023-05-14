import tensorflow as tf


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += tf.keras.backend.mean(tf.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # dr = dr.float()
        # dg = dg.float()
        r_loss = tf.keras.backend.mean((1 - dr) ** 2)
        g_loss = tf.keras.backend.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = tf.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, total_logdet, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    # z_p = z_p.float()
    # logs_q = logs_q.float()
    # m_p = m_p.float()
    # logs_p = logs_p.float()
    # z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * tf.exp(-2.0 * logs_p)
    kl = tf.reduce_sum(kl * z_mask)
    # add total_logdet (Negative LL)
    kl -= tf.reduce_sum(total_logdet)
    l = kl / tf.reduce_sum(z_mask)#tf.cast(z_mask,dtype=tf.float32))
    return l


def kl_loss_back(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    # z_p = z_p.float()
    # logs_q = logs_q.float()
    # m_p = m_p.float()
    # logs_p = logs_p.float()
    # z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * tf.exp(-2.0 * logs_p)
    kl = tf.keras.backend.sum(kl * z_mask)
    l = kl / tf.keras.backend.sum(z_mask)
    return l
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = tf.tanh(in_act[:, :n_channels_int, :])
    s_act = tf.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts